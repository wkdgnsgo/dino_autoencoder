
import torch
import torch.nn as nn
import timm

def build_mlp(proj_dim, hidden_dim, output_dim=None, activation=nn.SiLU()):
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
  
        super().__init__()

        self.dinov2 = dinov2_model
        for param in self.dinov2.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.dinov2.embed_dim

        self.bottleneck = build_mlp(
            proj_dim=self.embed_dim,      
            hidden_dim=mlp_hidden_dim,  
            output_dim=latent_dim,      
            activation=nn.SiLU()
        )

    
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False), 

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):

        features_dict = self.dinov2.forward_features(x)
        patch_tokens = features_dict['x_norm_patchtokens'] # Shape: (b, 256, 768)
        
        b, n, c = patch_tokens.shape
        h = w = int(n**0.5) 

        reshaped_tokens = patch_tokens.reshape(b, h, w, c) # (b, 16, 16, 768)
        latent_vector = self.bottleneck(reshaped_tokens)   # (b, 16, 16, 16)

        # decoder
        latent_vector_chw = latent_vector.permute(0, 3, 1, 2)
        reconstructed_image = self.decoder(latent_vector_chw)
        
        return reconstructed_image
