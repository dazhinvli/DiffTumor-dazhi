import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    Resized, RandFlipd, RandRotated, ToTensord
)
from monai.data import Dataset, DataLoader

class LoadImaged_BodyMap:
    """医学影像加载器（适配LiTS数据集）"""
    def __init__(self, image_only=True, roi_size=(96,96,96), a_min=-175, a_max=250):
        self.image_only = image_only
        self.roi_size = roi_size
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, data):
        # 手动实现窗宽窗位截断（替代 a_min/a_max 参数）
        data["image"] = torch.clamp(data["image"], self.a_min, self.a_max)
        return data    

    def get_transforms(self, is_train=True):
        """构建数据预处理流水线"""
        transforms = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            #NormalizeIntensityd(keys=["image"], a_min=self.a_min, a_max=self.a_max, b_min=-1.0, b_max=1.0),
            Resized(keys=["image"], spatial_size=self.roi_size, mode="trilinear", align_corners=False)
        ]
        
        # 训练阶段数据增强
        if is_train:
            transforms.extend([
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandRotated(keys=["image"], prob=0.5, range_x=10.0, range_y=10.0, range_z=10.0)
            ])
        
        # 转Tensor
        transforms.append(ToTensord(keys=["image"]))
        transforms.append(self)
        return Compose(transforms)

def get_train_val_dataloader(cfg, val_split=0.2):
    """获取训练/验证DataLoader"""
    # 1. 构建数据列表
    data_root = "/home/konglz/DiffTumor/data/"
    data_files = [f for f in os.listdir(data_root) if f.startswith("volume-") and f.endswith(".nii")]
    data_list = [{"image": os.path.join(data_root, f)} for f in data_files]
    
    # 2. 划分训练/验证集
    train_list, val_list = train_test_split(data_list, test_size=val_split, random_state=42)
    
    # 3. 构建Transform
    loader = LoadImaged_BodyMap(
        image_only=True,  # 硬编码为True
        roi_size=(96, 96, 96),  # 硬编码ROI尺寸
        a_min=-175,  # 硬编码CT窗宽
        a_max=250   # 硬编码CT窗位
    )
    train_transform = loader.get_transforms(is_train=True)
    val_transform = loader.get_transforms(is_train=False)
    
    # 4. 构建Dataset和DataLoader
    train_dataset = Dataset(data=train_list, transform=train_transform)
    val_dataset = Dataset(data=val_list, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    print(f"✅ 数据加载完成 | 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")
    return train_loader, val_loader
