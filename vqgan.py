import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf
from einops import rearrange

# taming-transformers에서 필요한 모듈 임포트
from taming.models.vqgan import VQModel
from taming.modules.diffusionmodules.model import ResnetBlock, AttnBlock

# -----------------------------------------------------------------------------
# 1. 이전 단계에서 정의한 Cross-Attention 모듈들
# -----------------------------------------------------------------------------

class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        inner_dim = n_heads * head_dim
        self.scale = head_dim ** -0.5
        self.n_heads = n_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # q, k, v를 head 수만큼 분할
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.n_heads), (q, k, v))

        # Attention 계산
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.n_heads)
        return self.to_out(out)

class CrossAttentionDecoderBlock(nn.Module):
    def __init__(self, decoder_dim, context_dim, n_heads=8, head_dim=64):
        super().__init__()
        self.attn = CrossAttentionLayer(decoder_dim, context_dim, n_heads, head_dim)
        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 4),
            nn.GELU(),
            nn.Linear(decoder_dim * 4, decoder_dim)
        )
        self.norm1 = nn.LayerNorm(decoder_dim)
        self.norm2 = nn.LayerNorm(decoder_dim)

    def forward(self, x, context):
        # x shape: (b, h*w, c), context shape: (b, n, d)
        x = x + self.attn(self.norm1(x), context=context)
        x = x + self.ffn(self.norm2(x))
        return x

# -----------------------------------------------------------------------------
# 2. VQGAN 모델을 로드하고, Cross-Attention을 적용한 디코더를 포함하는 메인 클래스
# -----------------------------------------------------------------------------

class VQGANDecoderWithCrossAttention(nn.Module):
    def __init__(self, vqgan_config_path, vqgan_ckpt_path, dino_dim=768):
        super().__init__()

        config = OmegaConf.load(vqgan_config_path)
        model = self._load_vqgan(config, vqgan_ckpt_path)
        
        # VQGAN의 주요 컴포넌트 추출 (인코더 제외)
        self.quantize = model.quantize
        self.quant_conv = model.quant_conv
        self.post_quant_conv = model.post_quant_conv
        
        # 코드북 차원 가져오기
        codebook_dim = config.model.params.embed_dim

        # DINOv2 특징을 코드북 차원으로 정렬하는 선형 레이어
        self.linear_align = nn.Linear(dino_dim, codebook_dim)

        # Cross-Attention이 적용된 새로운 디코더 생성
        self.decoder = self._build_cross_attention_decoder(model.decoder, dino_dim)
        
        # VQGAN 파라미터는 학습되지 않도록 고정
        self.quantize.eval()
        self.quant_conv.eval()
        self.post_quant_conv.eval()
        self.decoder.eval()
        for param in self.parameters():
            param.requires_grad = False
        
        # 학습시킬 부분만 requires_grad = True로 설정
        self.linear_align.requires_grad_(True)
        # 디코더의 cross_attn 부분만 학습 가능하게 변경
        for module in self.decoder.modules():
            if isinstance(module, CrossAttentionDecoderBlock):
                module.requires_grad_(True)
                module.train() # train 모드로 설정
        self.linear_align.train()


    def _load_vqgan(self, config, ckpt_path):
        """YAML 설정 파일과 체크포인트로부터 VQGAN 모델을 로드하는 헬퍼 함수"""
        config = OmegaConf.load(config_path)
        model = VQModel(**config.model.params)
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
        return model

    def _build_cross_attention_decoder(self, original_decoder, dino_dim):
        """기존 디코더 구조에 CrossAttention 블록을 삽입하는 함수"""
        new_blocks = nn.ModuleList()
        # original_decoder.body는 nn.Sequential
        for module in original_decoder.body:
            new_blocks.append(module)
            # AttnBlock(Self-Attention) 바로 뒤에 Cross-Attention 블록을 추가
            if isinstance(module, AttnBlock):
                # 현재 모듈의 채널 수를 가져옴
                decoder_dim = module.in_channels
                # CrossAttention 블록 추가
                new_blocks.append(CrossAttentionDecoderBlock(decoder_dim, dino_dim))
        
        # 기존 디코더의 다른 부분들도 유지
        original_decoder.body = new_blocks
        return original_decoder

    def forward(self, dino_features):
        """
        DINOv2 특징을 입력받아 이미지를 복원하고 양자화 손실을 반환하는 함수
        dino_features: (batch, num_patches, dino_dim), 예: (B, 256, 768)
        """
        # 1. DINO 특징을 코드북 차원으로 정렬
        # (B, 256, 768) -> (B, 256, 512)
        aligned_features = self.linear_align(dino_features)

        # 2. VQGAN이 기대하는 공간적 형태로 변경
        # (B, 256, 512) -> (B, 16, 16, 512) -> (B, 512, 16, 16)
        h = w = int(aligned_features.shape[1] ** 0.5)
        c = aligned_features.shape[2]
        x = rearrange(aligned_features, 'b (h w) c -> b c h w', h=h, w=w)

        # 3. 양자화 (Quantization)
        quant_in = self.quant_conv(x)
        z_q, quant_loss, (_, _, indices) = self.quantize(quant_in)

        # 4. Cross-Attention이 적용된 디코더로 복원
        quant_out = self.post_quant_conv(z_q)
        
        # 디코더 순회 (Cross-Attention 블록은 dino_features를 context로 사용)
        dec_state = quant_out
        for module in self.decoder.body:
            if isinstance(module, CrossAttentionDecoderBlock):
                # (b, c, h, w) -> (b, h*w, c) 형태로 변경하여 전달
                dec_state_rearranged = rearrange(dec_state, 'b c h w -> b (h w) c')
                # Cross-Attention 수행
                attn_out = module(dec_state_rearranged, context=dino_features)
                # 원래 형태로 복원
                dec_state = rearrange(attn_out, 'b (h w) c -> b c h w', h=h, w=w)
            else:
                dec_state = module(dec_state)
        
        # 최종 이미지 복원
        reconstructed_image = self.decoder.out(dec_state)

        return reconstructed_image, quant_loss

# -----------------------------------------------------------------------------
# 3. 사용 예시
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # ImageNet으로 사전학습된 VQGAN f16 모델의 경로
    # 이 파일들은 taming-transformers 레포지토리에서 다운로드 받아야 합니다.
    # https://omoro.com/models/
    VQGAN_CONFIG_PATH = "./logs/vqgan_imagenet_f16_1024/configs/model.yaml"
    VQGAN_CKPT_PATH = "./logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt"

    try:
        # 모델 인스턴스화
        model = VQGANDecoderWithCrossAttention(
            vqgan_config_path=VQGAN_CONFIG_PATH,
            vqgan_ckpt_path=VQGAN_CKPT_PATH,
            dino_dim=768 # DINOv2 Base 모델의 특징 차원
        )
        print("✅ VQGANDecoderWithCrossAttention 모델 생성 성공!")
        
        # 학습시킬 파라미터 수 확인
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🧠 학습 가능한 파라미터 수: {trainable_params:,}")

        # 가상 DINOv2 출력 생성
        dummy_dino_features = torch.randn(2, 256, 768) # (Batch, Patches, Dim)

        # 이미지 복원 실행
        restored_img = model(dummy_dino_features)

        print("⚡️ 이미지 복원 실행 완료!")
        print("DINO 입력 Shape:", dummy_dino_features.shape)
        print("복원된 이미지 Shape:", restored_img.shape) # 예: (2, 3, 256, 256)

    except FileNotFoundError:
        print("❌ 에러: VQGAN 설정 파일 또는 체크포인트 파일을 찾을 수 없습니다.")
        print(f"'{VQGAN_CONFIG_PATH}' 와 '{VQGAN_CKPT_PATH}' 경로를 확인해주세요.")