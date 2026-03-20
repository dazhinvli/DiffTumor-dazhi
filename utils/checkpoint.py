import os
import torch

def save_checkpoint(model, optimizer, epoch, loss, save_dir, name="vqgan"):
    """保存模型断点"""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    save_path = os.path.join(save_dir, f"{name}_epoch_{epoch}.pth")
    torch.save(checkpoint, save_path)
    print(f"✅ 模型已保存: {save_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """加载模型断点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"断点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["loss"]
    
    print(f"✅ 加载断点成功 | 起始轮次: {start_epoch} | 历史损失: {best_loss:.4f}")
    return model, optimizer, start_epoch, best_loss
