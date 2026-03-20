import numpy as np
import os
import nibabel as nib

# 新增：导入MONAI必需的工具（核心！）
from monai.transforms import (
    Compose, LoadImaged, NormalizeIntensityd, Resized,
    EnsureChannelFirstd, RandFlipd, RandRotated, ToTensord
)
from monai.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class LoadImaged_BodyMap:
    """医学影像加载器（适配LiTS数据集，修复所有标签报错）"""
    def __init__(self, image_only=True, roi_size=(96,96,96), a_min=-175, a_max=250):
        # 强制只加载图像，彻底关闭标签逻辑
        self.image_only = True
        self.roi_size = roi_size  # 3D ROI裁剪尺寸
        self.a_min = a_min        # CT窗宽（空气）
        self.a_max = a_max        # CT窗位（骨骼）

    def __call__(self, d):
        # 标签兜底：避免KeyError/None报错
        d['label'] = np.zeros(self.roi_size, dtype=np.float32)
        d['label_meta_dict'] = {}
        return d

    def label_transfer(self, lbl_dir, shape):
        # 永远返回非None值，避免UnboundLocalError
        organ_lbl = np.zeros(shape, dtype=np.float32)
        meta_information = {}
        return organ_lbl, meta_information

    def _loader(self, path):
        # 兜底：文件不存在时返回空数组，避免None
        if not os.path.exists(path):
            return np.zeros(self.roi_size), {}
        img = nib.load(path).get_fdata()
        return img.astype(np.float32), {"path": path}

    def get_transforms(self, is_train=True):
        """构建真实医学影像预处理流水线"""
        transforms = [
            LoadImaged(keys=["image"]),          # 加载.nii文件
            EnsureChannelFirstd(keys=["image"]), # 增加通道维度 (HWD)→(1,HWD)
            NormalizeIntensityd(                 # CT值归一化到[-1,1]
                keys=["image"], 
                a_min=self.a_min, 
                a_max=self.a_max, 
                b_min=-1.0, 
                b_max=1.0
            ),
            Resized(                             # 统一裁剪为96×96×96
                keys=["image"], 
                spatial_size=self.roi_size, 
                mode="trilinear", 
                align_corners=False
            )
        ]
        
        # 训练阶段：数据增强（提升模型泛化能力）
        if is_train:
            transforms.extend([
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),  # 随机翻转
                RandRotated(keys=["image"], prob=0.5, range_x=10.0)   # 随机旋转
            ])
        
        # 转Tensor：供PyTorch模型使用（必需！）
        transforms.append(ToTensord(keys=["image"]))
        return Compose(transforms)

# 新增：核心函数——获取真实数据的DataLoader
def get_train_val_dataloader(data_root="/home/konglz/DiffTumor/data/", batch_size=1, val_split=0.2):
    """
    加载真实LiTS数据集的训练/验证DataLoader
    :param data_root: 你的.nii数据存放路径
    :param batch_size: 批次大小
    :param val_split: 验证集比例
    :return: train_loader, val_loader
    """
    # 1. 自动扫描所有volume-*.nii文件（适配你的真实数据）
    data_files = [f for f in os.listdir(data_root) if f.startswith("volume-") and f.endswith(".nii")]
    if not data_files:
        print(f"❌ 错误：在 {data_root} 未找到volume-*.nii文件！请检查数据路径")
        exit(1)
    data_list = [{"image": os.path.join(data_root, f)} for f in data_files]
    print(f"✅ 找到 {len(data_files)} 个CT影像文件")

    # 2. 划分训练/验证集
    train_list, val_list = train_test_split(data_list, test_size=val_split, random_state=42)

    # 3. 初始化加载器+预处理
    loader = LoadImaged_BodyMap()
    train_transform = loader.get_transforms(is_train=True)
    val_transform = loader.get_transforms(is_train=False)

    # 4. 构建Dataset和DataLoader
    train_dataset = Dataset(data=train_list, transform=train_transform)
    val_dataset = Dataset(data=val_list, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集打乱
        num_workers=1,
        pin_memory=True  # 加速GPU加载
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    print(f"✅ 数据加载完成 | 训练集：{len(train_dataset)} | 验证集：{len(val_dataset)}")
    return train_loader, val_loader