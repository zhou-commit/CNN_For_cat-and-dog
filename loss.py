import torch


class CustomMSEloss(torch.nn.Module):
    def __init__(self):
        super(CustomMSEloss, self).__init__()

    def forward(self, output, target):
        return torch.mean((output - target) ** 2)
        #均方误差损失函数，计算输出和目标之间的均方误差


def MY_MSEloss(output, target):
    return  torch.mean((output - target) ** 2)


# 创建模型和数据
torch.manual_seed(0)
model = torch.nn.Linear(1, 1)  # 一个输入特征，一个输出特征
# criterion_custom = CustomMSEloss()
criterion_torch = torch.nn.MSELoss()

# 随机生成一些数据
inputs = torch.randn(10, 1)  # 10个样本，1个特征
targets = torch.randn(10, 1)


# 使用自定义损失函数计算损失
# loss_custom = criterion_custom(model(inputs), targets)
loss_custom =MY_MSEloss(model(inputs), targets)
print("自定义MSE的损失:", loss_custom.item())

# 使用内置的PyTorch损失函数计算损失
loss_builtin = criterion_torch(model(inputs), targets)
print("官方书写的MSE的损失:", loss_builtin.item())
