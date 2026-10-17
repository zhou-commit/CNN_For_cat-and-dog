import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 数据集根目录（相对脚本所在目录解析，避免受运行目录影响，与 triain.py 保持一致）
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cat-dog")
train_path = os.path.join(DATA_ROOT, "train")

# 预处理：resize 到 224x224（匹配模型输入）；标准化参数与 triain.py 完全一致（ImageNet 统计量），
# 保证演示用的加载流程与训练流程看到同一份数据分布
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                      # PIL(HWC uint8) -> CHW float tensor，并除以 255
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# ImageFolder 自动遍历子文件夹：标签由文件夹名决定（按字母序 cat=0, dog=1）
train_dataset = ImageFolder(root=train_path, transform=data_transform)

print("类别 -> 标签映射:", train_dataset.class_to_idx)  # {'cat': 0, 'dog': 1}
print("样本总数:", len(train_dataset))

datas = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
# batch_size=6：每次取 6 个样本，共 10 个Batch,shuffle=True：打乱数据顺序，num_workers=0：不使用多线程
for i, (data, labels) in enumerate(datas):
    print("第 {} 个Batch \n 数据形状: {} \n 标签: {}".format(i, data.shape, labels))
    if i >= 2:
        break

# 验证集/测试集目录结构相同，换 root 即可复用
val_path = os.path.join(DATA_ROOT, "validate")
test_path = os.path.join(DATA_ROOT, "test")
# val_dataset = ImageFolder(root=val_path, transform=data_transform)
# test_dataset = ImageFolder(root=test_path, transform=data_transform)
