
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb
import os
from accelerate import Accelerator

# Hugging Face datasets 라이브러리 임포트
from datasets import load_dataset

# model.py에서 정의한 모델 클래스 임포트
from model import DINOv2Autoencoder, build_mlp

# --- 1. 학습 설정 ---
# 하이퍼파라미터
BATCH_SIZE = 1024
NUM_EPOCHS = 1000
LEARNING_RATE = 8e-3
LATENT_DIM = 16
MLP_HIDDEN_DIM = 256
DINOV2_MODEL_NAME = 'dinov2_vitb14' # DINOv2 모델 이름 (embed_dim=768)

# 디바이스 설정
accelerator = Accelerator()

# WandB 초기화
if accelerator.is_main_process:
    wandb.init(project="dinov2-autoencoder-imagenet", group="DINOv2_16_dim_Bottleneck", config={
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "latent_dim": LATENT_DIM,
        "mlp_hidden_dim": MLP_HIDDEN_DIM,
        "dinov2_model": DINOV2_MODEL_NAME,
        "reconstruction_method": "DINOv2_16_dim_Bottleneck",
    })



# --- 2. 데이터 로드 및 전처리 ---
# DINOv2는 ImageNet 정규화를 사용합니다.
# mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Hugging Face datasets에서 ImageNet-1k 로드
if accelerator.is_main_process:
    print("Loading ImageNet-1k dataset from Hugging Face Hub...")
# 'validation' split이 'val'로 명명되어 있을 수 있으니 확인 필요
# ILSVRC/imagenet-1k는 'train'과 'validation' 스플릿을 가집니다.
imagenet1k = load_dataset("ILSVRC/imagenet-1k")
if accelerator.is_main_process:
    print("Dataset loaded.")

# 데이터셋에 변환 함수 적용
# datasets 라이브러리는 PIL Image를 반환하므로, torchvision.transforms를 적용합니다.
def apply_transforms_to_dataset(examples):
    # 'image'는 datasets의 기본 이미지 컬럼 이름입니다.
    # .convert("RGB")는 혹시 모를 RGBA 등을 RGB로 변환합니다.
    examples['pixel_values'] = [transform(image.convert("RGB")) for image in examples['image']]
    return examples

train_dataset_hf = imagenet1k['train']
val_dataset_hf = imagenet1k['validation']

# set_transform을 사용하여 데이터셋에 변환 함수를 적용합니다.
# 이 함수는 배치 단위로 호출되며, 'pixel_values' 키에 변환된 텐서를 저장합니다.
train_dataset_hf.set_transform(apply_transforms_to_dataset)
val_dataset_hf.set_transform(apply_transforms_to_dataset)

# DataLoader 생성
# DataLoader는 'pixel_values' 키를 가진 딕셔너리를 반환할 것입니다.
# collate_fn을 사용하여 딕셔너리에서 'pixel_values'만 추출하도록 합니다.
def collate_fn(batch):
    # batch는 list of dicts (e.g., [{'pixel_values': tensor1}, {'pixel_values': tensor2}, ...])
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    return pixel_values

train_loader = DataLoader(train_dataset_hf, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset_hf, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

# --- 3. 모델, 손실 함수, 옵티마이저 초기화 ---
if accelerator.is_main_process:
    print(f"Loading DINOv2 model: {DINOV2_MODEL_NAME} from torch.hub...")
# DINOv2 모델 로드 (torch.hub.load 사용)
dinov2_hub_model = torch.hub.load('facebookresearch/dinov2', DINOV2_MODEL_NAME)
dinov2_hub_model.eval().to(accelerator.device) # DINOv2는 학습하지 않으므로 eval 모드로 고정, 현재 장치로 이동
if accelerator.is_main_process:
    print("DINOv2 model loaded.")

accelerator.wait_for_everyone()

model = DINOv2Autoencoder(
    dinov2_model=dinov2_hub_model,
    latent_dim=LATENT_DIM,
    mlp_hidden_dim=MLP_HIDDEN_DIM
)

# 학습 가능한 파라미터 수 확인 (DINOv2 제외)
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if accelerator.is_main_process:
    print(f"Total trainable parameters in Autoencoder (Bottleneck + Decoder): {total_trainable_params:,}")
    wandb.config.update({"trainable_params": total_trainable_params})

criterion = nn.MSELoss() # 재구성 손실 (Mean Squared Error)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

if accelerator.is_main_process:
    print(f"Train dataset size: {len(train_dataset_hf)}")
if accelerator.is_main_process:
    print(f"Validation dataset size: {len(val_dataset_hf)}")

# --- 4. 학습 및 검증 루프 ---
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    # --- 학습 단계 ---
    model.train()
    train_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
    for batch_idx, images in enumerate(train_pbar): # collate_fn 덕분에 images만 바로 받음

        optimizer.zero_grad()
        reconstructed_images = model(images)
        loss = criterion(reconstructed_images, images)
        accelerator.backward(loss)
        optimizer.step()

        train_loss += loss.item()
        train_pbar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    if accelerator.is_main_process:
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
        if accelerator.is_main_process:
            print(f"Logging train_loss to WandB for epoch {epoch+1}")
            wandb.log({"train_loss": avg_train_loss}, step=epoch)

    # --- 검증 단계 ---
    model.eval()
    val_loss = 0.0
    # WandB에 로깅할 이미지 샘플을 저장할 리스트
    logged_images = [] 
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", leave=False)
        for batch_idx, images in enumerate(val_pbar): # collate_fn 덕분에 images만 바로 받음
            reconstructed_images = model(images)
            loss = criterion(reconstructed_images, images)
            val_loss += loss.item()
            val_pbar.set_postfix(loss=loss.item())

            # 첫 번째 배치에서 원본 및 복원 이미지 샘플 로깅
            if batch_idx == 0 and epoch % 1 == 0: # 매 에폭 첫 배치만 로깅
                # 이미지 정규화 해제 (WandB 로깅을 위해)
                # DINOv2 정규화: pixel = (pixel - mean) / std
                # 역변환: pixel = pixel * std + mean
                mean = torch.tensor([0.485, 0.456, 0.406]).to(accelerator.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).to(accelerator.device).view(1, 3, 1, 1)

                original_display = (images * std + mean).clamp(0, 1)
                reconstructed_display = (reconstructed_images * std + mean).clamp(0, 1)

                # 몇 개의 이미지 샘플만 로깅
                num_samples_to_log = min(4, images.shape[0])
                for i in range(num_samples_to_log):
                    logged_images.append(wandb.Image(original_display[i], caption=f"Epoch {epoch+1} - Original"))
                    logged_images.append(wandb.Image(reconstructed_display[i], caption=f"Epoch {epoch+1} - Reconstructed"))
        
    avg_val_loss = val_loss / len(val_loader)
    if accelerator.is_main_process:
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        if accelerator.is_main_process:
            print(f"Logging val_loss to WandB for epoch {epoch+1}")
            wandb.log({"val_loss": avg_val_loss}, step=epoch)
        
        # WandB에 이미지 로깅
        if logged_images:
            if accelerator.is_main_process:
                print(f"Logging input_output_images to WandB for epoch {epoch+1}")
                wandb.log({"input_output_images": logged_images}, step=epoch)

        # 스케줄러 업데이트
        if accelerator.is_main_process:
            scheduler.step(avg_val_loss)

        # 모델 저장 (validation loss 기준)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_dir = "./saved_models"
            os.makedirs(output_dir, exist_ok=True)
            accelerator.save_state(os.path.join(output_dir, f"best_model_epoch_{epoch+1}.pth"))
            print(f"Saved best model at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

        # 매 에폭마다 모델 저장
        epoch_output_dir = "./saved_models/epochs"
        os.makedirs(epoch_output_dir, exist_ok=True)
        accelerator.save_state(os.path.join(epoch_output_dir, f"model_epoch_{epoch+1}.pth"))
        print(f"Saved model for epoch {epoch+1}.")

if accelerator.is_main_process:
    print("Training finished!")
    wandb.finish()
