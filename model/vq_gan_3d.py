import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """VQ-VAE 核心量化模块（带EMA更新）"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        # 初始化码本
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # EMA参数
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.clone(self.embedding.weight))

    def forward(self, z):
        # 调整形状: (B, C, D, H, W) → (B, D, H, W, C) → (B*D*H*W, C)
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        
        # 计算距离: (N, M) = (N, D) @ (D, M)
        dist = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight**2, dim=1) - \
               2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        # 硬量化：选择最近的码本向量
        encoding_indices = torch.argmin(dist, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        
        # 计算损失（承诺损失 + 码本损失）
        loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)
        
        # EMA更新码本（仅训练阶段）
        if self.training:
            # 更新EMA计数
            one_hot = F.one_hot(encoding_indices, self.num_embeddings).type(z.dtype)
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * torch.sum(one_hot, dim=0)
            
            # 平滑
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            
            # 更新码本
            dw = torch.matmul(one_hot.t(), z_flattened)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
            self.embedding.weight.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)
        
        # 直通估计（STE）：梯度回传给z，值用量化后的z_q
        z_q = z + (z_q - z).detach()
        
        # 还原形状: (B, D, H, W, C) → (B, C, D, H, W)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        return z_q, loss, encoding_indices

class VQGAN3D(nn.Module):
    """3D VQGAN（用于医学影像合成）"""
    def __init__(self, in_channels=1, latent_dim=128, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        
        # 编码器（下采样）
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            
            nn.Conv3d(128, latent_dim, kernel_size=1, stride=1, padding=0)
        )
        
        # 矢量量化模块
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        
        # 解码器（上采样）
        self.decoder = nn.Sequential(
            nn.Conv3d(latent_dim, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # 输出归一化到[-1,1]
        )

    def forward(self, x):
        # 编码
        z = self.encoder(x)
        # 量化
        z_q, vq_loss, indices = self.quantizer(z)
        # 解码
        x_recon = self.decoder(z_q)
        # 计算重建损失
        recon_loss = F.mse_loss(x_recon, x)
        # 总损失
        total_loss = recon_loss + vq_loss
        return x_recon, total_loss, recon_loss, vq_loss
