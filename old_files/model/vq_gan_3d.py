import torch
import torch.nn as nn

class VQGAN3D(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv3d(64, latent_dim, 3, 2, 1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 64, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, in_channels, 3, 2, 1, output_padding=1),
            nn.Tanh()
        )
        self.quantize = nn.Identity()

    def forward(self, x):
        z = self.encoder(x)
        z_q = self.quantize(z)
        x_recon = self.decoder(z_q)
        return x_recon, z, z_q
