
import torch
import torch.nn as nn
import timm

def build_mlp(proj_dim, hidden_dim, output_dim=None, activation=nn.SiLU()):
    """
    주어진 차원에 따라 MLP를 구성합니다.
    """
    if output_dim is None:
        output_dim = hidden_dim
    return nn.Sequential(
        nn.Linear(proj_dim, hidden_dim),
        activation,
        nn.Linear(hidden_dim, hidden_dim),
        activation,
        nn.Linear(hidden_dim, output_dim)
    )

class DINOv2Autoencoder(nn.Module):
    def __init__(self, dinov2_model, latent_dim=16, mlp_hidden_dim=256):
        """
        DINOv2를 인코더로, MLP를 병목으로 사용하는 오토인코더.

        Args:
            dinov2_model (torch.nn.Module): 미리 로드된 DINOv2 모델 인스턴스.
            latent_dim (int): 최종 잠재 벡터의 채널 차원.
            mlp_hidden_dim (int): 병목 MLP의 중간 레이어 차원.
        """
        super().__init__()

        # 1. 인코더 (Encoder)
        self.dinov2 = dinov2_model
        # DINOv2의 가중치가 학습 중에 업데이트되지 않도록 고정
        for param in self.dinov2.parameters():
            param.requires_grad = False
            
        # DINOv2 모델의 임베딩 차원 확인
        # timm 모델의 경우 embed_dim 속성을 통해 접근 가능
        self.embed_dim = self.dinov2.embed_dim

        # 2. 압축 (Bottleneck) - MLP 구조 사용
        self.bottleneck = build_mlp(
            proj_dim=self.embed_dim,      # 입력 차원 (예: 768)
            hidden_dim=mlp_hidden_dim,   # 중간 차원 (예: 256)
            output_dim=latent_dim,       # 출력 차원 (예: 16)
            activation=nn.SiLU()
        )

        # 3. 디코더 (Decoder)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 16x16 -> 32x32

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 32x32 -> 64x64

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 64x64 -> 128x128
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False), # 128x128 -> 224x224

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid() # 최종 출력 픽셀 값을 0과 1 사이로 조정
        )

    def forward(self, x):
        # --- 인코더 ---
        # DINOv2를 통과시켜 피처 추출
        # forward_features는 피처 딕셔너리를 반환합니다.
        features_dict = self.dinov2.forward_features(x)
        patch_tokens = features_dict['x_norm_patchtokens'] # Shape: (b, 256, 768)
        
        b, n, c = patch_tokens.shape
        h = w = int(n**0.5) # 256 -> 16

        # --- 차원 변경 및 압축 ---
        reshaped_tokens = patch_tokens.reshape(b, h, w, c) # (b, 16, 16, 768)
        latent_vector = self.bottleneck(reshaped_tokens)   # (b, 16, 16, 16)

        # --- 디코더 ---
        # PyTorch의 Conv 레이어는 채널이 두 번째 차원에 와야 합니다 (NCHW).
        latent_vector_chw = latent_vector.permute(0, 3, 1, 2)
        reconstructed_image = self.decoder(latent_vector_chw)
        
        return reconstructed_image
