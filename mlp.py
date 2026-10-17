import torch
import torch.nn as nn
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(3 * 224 * 224, 128)  # 输入大小为3*224*224，输出大小为128
        self.fc2 = nn.Linear(128, 64)  # 输入大小为128，输出大小为64
        self.fc3 = nn.Linear(64, 10)  # 输入大小为64，输出大小为10，假设有10个类别

    def forward(self, x):
        x = x.view(-1, 3 * 224 * 224)  # 将输入展平
        x = torch.relu(self.fc1(x))  # 第一个全连接层并应用ReLU激活函数
        x = torch.relu(self.fc2(x))  # 第二个全连接层并应用ReLU激活函数
        x = self.fc3(x)  # 输出层
        return x

# 创建模型
model = SimpleMLP()

# 打印模型结构
print(model)

# mlp即多层感知器，nn.linear即全连接层
