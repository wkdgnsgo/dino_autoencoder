
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
import lpips

from datasets import load_dataset

from vqgan import VQGANDecoderWithCrossAttention


BATCH_SIZE = 1024
NUM_EPOCHS = 1000
LEARNING_RATE = 8e-3
DINOV2_MODEL_NAME = 'dinov2_vitb14'
VQGAN_CONFIG_PATH = "./logs/vqgan_imagenet_f16_16384/configs/model.yaml"
VQGAN_CKPT_PATH = "./logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt"


accelerator = Accelerator()

if accelerator.is_main_process:
    wandb.init(project="dino-vqgan-reconstruction", group="DINOv2_VQGAN_CrossAttention", config={
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "dinov2_model": DINOV2_MODEL_NAME,
        "vqgan_config": VQGAN_CONFIG_PATH,
    })


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

if accelerator.is_main_process:
    print("Loading ImageNet-1k dataset from Hugging Face Hub...")
imagenet1k = load_dataset("ILSVRC/imagenet-1k")
if accelerator.is_main_process:
    print("Dataset loaded.")

def apply_transforms_to_dataset(examples):
    examples['pixel_values'] = [transform(image.convert("RGB")) for image in examples['image']]
    return examples

train_dataset_hf = imagenet1k['train']
val_dataset_hf = imagenet1k['validation']

train_dataset_hf.set_transform(apply_transforms_to_dataset)
val_dataset_hf.set_transform(apply_transforms_to_dataset)

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    return pixel_values

train_loader = DataLoader(train_dataset_hf, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset_hf, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

if accelerator.is_main_process:
    print(f"Loading DINOv2 model: {DINOV2_MODEL_NAME} from torch.hub...")

dinov2_hub_model = torch.hub.load('facebookresearch/dinov2', DINOV2_MODEL_NAME)
dinov2_hub_model.eval().to(accelerator.device)

if accelerator.is_main_process:
    print("DINOv2 model loaded.")

accelerator.wait_for_everyone()

model = VQGANDecoderWithCrossAttention(
    vqgan_config_path=VQGAN_CONFIG_PATH,
    vqgan_ckpt_path=VQGAN_CKPT_PATH,
    dino_dim=dinov2_hub_model.embed_dim
)

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if accelerator.is_main_process:
    print(f"Total trainable parameters in VQGAN (linear_align + CrossAttention): {total_trainable_params:,}")
    wandb.config.update({"trainable_params": total_trainable_params})

# Loss functions
loss_fn_alex = lpips.LPIPS(net='alex').to(accelerator.device)
l1_loss = nn.L1Loss()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

if accelerator.is_main_process:
    print(f"Train dataset size: {len(train_dataset_hf)}")
if accelerator.is_main_process:
    print(f"Validation dataset size: {len(val_dataset_hf)}")


best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):

    model.train()
    train_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
    for batch_idx, images in enumerate(train_pbar):

        optimizer.zero_grad()
        
        with torch.no_grad():
            features_dict = dinov2_hub_model.forward_features(images)
            dino_features = features_dict['x_norm_patchtokens']

        reconstructed_images, quant_loss = model(dino_features)
        
        # VQGAN Loss Calculation
        rec_loss = l1_loss(reconstructed_images, images)
        p_loss = loss_fn_alex(reconstructed_images, images).mean()
        
        total_loss = rec_loss + quant_loss + p_loss
        
        accelerator.backward(total_loss)
        optimizer.step()

        train_loss += total_loss.item()
        train_pbar.set_postfix(loss=total_loss.item())

    avg_train_loss = train_loss / len(train_loader)
    if accelerator.is_main_process:
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss, "rec_loss": rec_loss.item(), "p_loss": p_loss.item(), "quant_loss": quant_loss.item()}, step=epoch)


    model.eval()
    val_loss = 0.0

    logged_images = []

    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", leave=False)
        for batch_idx, images in enumerate(val_pbar):
            features_dict = dinov2_hub_model.forward_features(images)
            dino_features = features_dict['x_norm_patchtokens']
            
            reconstructed_images, quant_loss = model(dino_features)
            
            rec_loss = l1_loss(reconstructed_images, images)
            p_loss = loss_fn_alex(reconstructed_images, images).mean()
            
            total_loss = rec_loss + quant_loss + p_loss
            val_loss += total_loss.item()
            val_pbar.set_postfix(loss=total_loss.item())

            if batch_idx == 0 and epoch % 1 == 0:
                mean = torch.tensor([0.485, 0.456, 0.406]).to(accelerator.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).to(accelerator.device).view(1, 3, 1, 1)

                original_display = (images * std + mean).clamp(0, 1)
                reconstructed_display = (reconstructed_images * std + mean).clamp(0, 1)

                num_samples_to_log = min(4, images.shape[0])
                for i in range(num_samples_to_log):
                    logged_images.append(wandb.Image(original_display[i], caption=f"Epoch {epoch+1} - Original"))
                    logged_images.append(wandb.Image(reconstructed_display[i], caption=f"Epoch {epoch+1} - Reconstructed"))

    avg_val_loss = val_loss / len(val_loader)
    if accelerator.is_main_process:
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss}, step=epoch)

        if logged_images:
            wandb.log({"input_output_images": logged_images}, step=epoch)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_dir = "./saved_models"
            os.makedirs(output_dir, exist_ok=True)
            accelerator.save_state(os.path.join(output_dir, f"best_model_epoch_{epoch+1}.pth"))
            print(f"Saved best model at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

        epoch_output_dir = "./saved_models/epochs"
        os.makedirs(epoch_output_dir, exist_ok=True)
        accelerator.save_state(os.path.join(epoch_output_dir, f"model_epoch_{epoch+1}.pth"))
        print(f"Saved model for epoch {epoch+1}.")

if accelerator.is_main_process:
    print("Training finished!")
    wandb.finish()
