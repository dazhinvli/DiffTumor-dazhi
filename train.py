import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from dataset.dataloader import get_train_val_dataloader
from model.vq_gan_3d import VQGAN3D
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint

@hydra.main(version_base=None, config_path="config", config_name="base_cfg")
def main(cfg: DictConfig):
    # 1. 初始化
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger = setup_logger(os.getcwd(), "vqgan_train")
    logger.info("="*60)
    logger.info("🚀 启动 DiffTumor VQGAN 3D 训练")
    logger.info(f"📌 配置: {OmegaConf.to_yaml(cfg)}")
    logger.info(f"📌 设备: {device}")
    logger.info("="*60)

    # 2. 加载数据（此时 dataloader 内部已硬编码路径）
    train_loader, val_loader = get_train_val_dataloader(cfg.dataset, val_split=cfg.val.val_split)

    # 3. 初始化模型/优化器
    model = VQGAN3D(
        in_channels=1,  # 硬编码为1（医学影像单通道）
        latent_dim=128,  # 硬编码 latent_dim
        num_embeddings=512,  # 硬编码码本大小
        commitment_cost=0.25  # 硬编码承诺损失权重
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )

    # 4. 训练循环
    best_val_loss = float("inf")
    for epoch in range(cfg.train.epochs):
        # ---------------------- 训练阶段 ----------------------
        model.train()
        train_total_loss = 0.0
        train_recon_loss = 0.0
        train_vq_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            x = batch["image"].to(device)
            optimizer.zero_grad()

            # 前向传播
            x_recon, total_loss, recon_loss, vq_loss = model(x)
            
            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 累计损失
            train_total_loss += total_loss.item()
            train_recon_loss += recon_loss.item()
            train_vq_loss += vq_loss.item()

        # 计算平均训练损失
        train_total_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_vq_loss /= len(train_loader)

        # ---------------------- 验证阶段 ----------------------
        if (epoch + 1) % cfg.val.val_freq == 0:
            model.eval()
            val_total_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["image"].to(device)
                    _, total_loss, _, _ = model(x)
                    val_total_loss += total_loss.item()
            
            val_total_loss /= len(val_loader)
            # 更新最佳损失
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                save_checkpoint(model, optimizer, epoch, best_val_loss, "./checkpoints", "vqgan_best")

        # ---------------------- 日志/保存 ----------------------
        # 打印日志
        if (epoch + 1) % cfg.train.log_freq == 0:
            log_str = f"Epoch [{epoch+1}/{cfg.train.epochs}] | "
            log_str += f"Train Loss: {train_total_loss:.4f} (Recon: {train_recon_loss:.4f}, VQ: {train_vq_loss:.4f}) | "
            if (epoch + 1) % cfg.val.val_freq == 0:
                log_str += f"Val Loss: {val_total_loss:.4f} | Best Val Loss: {best_val_loss:.4f}"
            logger.info(log_str)
        
        # 保存模型
        if (epoch + 1) % cfg.train.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, train_total_loss, "./checkpoints", "vqgan")

    # 5. 训练完成
    logger.info("="*60)
    logger.info("🎉 训练完成！")
    logger.info(f"🏆 最佳验证损失: {best_val_loss:.4f}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
