# 导入需要的库
import torch
import os
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets

'''
1. 设置随机种子
随机种子是用来保证结果可以复现
'''
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU随机种子确定
    torch.cuda.manual_seed(seed)  # GPU随机种子确定
    torch.cuda.manual_seed_all(seed)  # 所有的GPU设置种子
    random.seed(seed)

    torch.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭
    torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法

set_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
2. 定义构造数据集的方法
'''
# 创建数据转换操作（验证/测试：仅 resize + 标准化）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 将图像数据转换为Tensor格式
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 标准化图像数据
])

# 训练时额外做数据增强，缓解小数据集过拟合
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 数据集根目录（相对脚本所在目录解析，避免受运行目录影响）
# cat-dog 已按 train/validate/test 三个子目录分好，每个子目录下有 cat、dog 两类
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cat-dog")
train_dir = os.path.join(DATA_ROOT, "train")
val_dir = os.path.join(DATA_ROOT, "validate")
test_dir = os.path.join(DATA_ROOT, "test")

# 分别构建三个数据集（ImageFolder 会按文件夹名自动编号：cat=0, dog=1）
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# 打印类别到标签的映射，方便推理时对照
print("类别映射 class_to_idx:", train_dataset.class_to_idx)

'''
3.定义模型的训练流程（同时定义测试流程）
'''
# 训练模型函数
def train_model(model, dataloader, criterion, optimizer, val_loader=None, num_epochs=10):
    # 保存模型的目录（项目内 ./checkpoints，不存在则自动创建）
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用tqdm显示训练进度条
        with tqdm(dataloader, unit="batch") as tepoch:
            # 设置描述进度条内容
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs} [train]")
            for i, (inputs, labels) in enumerate(tepoch):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  # 梯度清零
                outputs = model(inputs)  # 前向传播计算模型输出
                loss = criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播计算梯度
                optimizer.step()  # 更新模型参数

                running_loss += loss.item()  # 累加损失值
                _, predicted = outputs.max(1)  # 获取预测结果
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                accuracy = 100. * correct / total
                tepoch.set_postfix(loss=running_loss / (i + 1), accuracy=accuracy)  # 更新进度条显示
                tepoch.update()

        # 每个 epoch 结束后在验证集上评估，并保存模型
        if val_loader is not None:
            val_acc = evaluate(model, val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}  train_loss={running_loss/(i+1):.4f}  "
                  f"train_acc={accuracy:.2f}%  val_acc={val_acc:.2f}%")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model, os.path.join(ckpt_dir, 'best.pt'))  # 只保留验证集上最优的模型
        else:
            print(f"Epoch {epoch+1}/{num_epochs}  train_loss={running_loss/(i+1):.4f}  train_acc={accuracy:.2f}%")

        torch.save(model, os.path.join(ckpt_dir, 'cp{}.pt'.format(epoch)))  # 保存每一个epoch的模型


def evaluate(model, dataloader):
    """在验证集上评估，返回准确率(%)"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total if total else 0.0

def test_model(model, dataloader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # 前向传播计算模型输出
            _, predicted = outputs.max(1)  # 获取预测结果
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

'''
4. 构建模型 
'''
# 加载预训练的ResNet-18模型（torchvision>=0.13 推荐用 weights 参数，pretrained 已废弃）
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 获取ResNet-18模型的最后一层全连接层的输入特征维度
num_features = resnet18.fc.in_features
# 替换最后一层全连接层，使其适应猫狗二分类（cat=0, dog=1）
resnet18.fc = torch.nn.Linear(num_features, 2)


'''
5. 定义数据集加载的方式
'''
# 冒烟测试开关：置 True 时只取少量样本 + 2 个 epoch，256张图，快速验证整条流程能否跑通
SMOKE_TEST = False
SMOKE_N = 256

if SMOKE_TEST:
    train_data = Subset(train_dataset, range(min(SMOKE_N, len(train_dataset))))
    val_data = Subset(val_dataset, range(min(SMOKE_N // 2, len(val_dataset))))
else:
    train_data = train_dataset
    val_data = val_dataset

# 创建训练集和验证集的数据加载器
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=False, num_workers=2)

val_loader = DataLoader(val_data, batch_size=64, shuffle=False, drop_last=False, num_workers=2)

# 测试集加载器（来自动 cat-dog/test，用于最终评估）
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=2)


'''
6. 定义损失函数和优化器
'''
# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = torch.optim.Adam(resnet18.parameters(), lr=0.001)  # 使用Adam优化器

if __name__ == '__main__':
    # 启动训练
    resnet18.to(device=device)
    epochs = 2 if SMOKE_TEST else 10
    # 冒烟测试只用2个epoch，完整测试用10个epoch
    train_model(resnet18, train_loader, criterion, optimizer, val_loader=val_loader, num_epochs=epochs)

    # 训练完成后在独立测试集上评估
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    best_model = torch.load(os.path.join(ckpt_dir, 'best.pt'), weights_only=False).to(device=device)
    print("===== 在 cat-dog/test 上评估最优模型 =====")
    test_model(best_model, test_loader)
    
