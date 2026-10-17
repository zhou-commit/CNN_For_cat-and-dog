# class CustomOptimizer:
#     def __init__(self, learning_rate=0.01):
#         self.learning_rate = learning_rate
#
#     def update(self, params, grads):
#         for param, grad in zip(params, grads):
#             param -= self.learning_rate * grad
#
#
# class LinearRegression:
#     def __init__(self):
#         self.W = 0.0  # 初始化权重为 0.0
#         self.b = 0.0  # 初始化偏置为 0.0
#
#     def forward(self, X):
#         return [self.W * x + self.b for x in X]
#
#     def backward(self, X, y):
#         # 初始化梯度
#         dW = 0.0
#         db = 0.0
#         n = len(X)
#
#         # 计算梯度的累积值
#         for xi, yi in zip(X, y):
#             dW += 2 * xi * (self.W * xi + self.b - yi)
#             db += 2 * (self.W * xi + self.b - yi)
#
#         return [dW / n, db / n]
#
#
# # 初始化自定义优化器
# optimizer = CustomOptimizer(learning_rate=0.01)
#
# # 初始化线性回归模型
# model = LinearRegression()
#
# # 示例数据
# X_train = [1, 2, 3, 4, 5]
# y_train = [2, 4, 6, 8, 10]
#
# # 训练模型
# epochs = 100
# for epoch in range(epochs):
#     # 前向传播
#     y_pred = model.forward(X_train)
#
#     # 计算损失
#     loss = sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y_train)) / len(y_train)
#
#     # 打印损失
#     if epoch % 10 == 0:
#         print(f'Epoch {epoch}, Loss: {loss}')
#
#     # 反向传播
#     grads = model.backward(X_train, y_train)
#
#     params = [model.W, model.b]
#     optimizer.update(params, params)
#     model.W = params[0]
#     model.b = params[1]
#
#     # # 更新模型参数
#     # model.W -= optimizer.learning_rate * grads[0]
#     # model.b -= optimizer.learning_rate * grads[1]
#
# # 打印训练后的参数
# print(f'W: {model.W}, b: {model.b}')

import torch
import torch.nn as nn
import numpy as np

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        # 初始化SGD优化器，SGD拥有出色的泛化能力，但lr难以调控
        super(SGD, self).__init__(params, defaults={'lr': lr})
        # 定义一个空字典，用于存储优化器的状态信息（在这个简化的实现中未被使用）
        self.state = dict()
        # 打印参数组信息（在这个简化的实现中仅用于调试）
        print(self.param_groups)
        # 遍历每个参数组
        for group in self.param_groups:
            # 遍历参数组中的每个参数
            for p in group['params']:
                # 为每个参数创建一个空字典，用于存储其状态信息（在这个简化的实现中未被使用）
                self.state[p] = dict()

    def step(self):
        # 遍历每个参数组，step（）方法用于更新模型参数，训练循环里反复调用
        for group in self.param_groups:
            # 打印当前参数组的信息（在这个简化的实现中仅用于调试）
            print(group)
            # 遍历当前参数组中的每个参数
            for p in group['params']:
                # 如果参数的梯度为None，则跳过该参数的更新，因为梯度为0代表已经找到了使参数最小的值
                if p.grad is None:
                    continue
                # 获取参数的梯度值
                d_p = p.grad.data
                # 利用随机梯度下降的更新规则更新参数的值（求最小值）
                p.data -= group['lr'] * d_p
                # 新参数 = 旧参数 - 学习率 * 梯度，梯度下降的优化算法


# 创建一个简单的线性回归模型
class LinearRegression(nn.Module):
    # 继承 PyTorch 的 nn.Module——所有神经网络模型的基类。继承它才能拥有参数管理（.parameters()）、自动求导、.to(device) 迁移等能力。
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
        # 创建一个全连接层，输入 1 个特征、输出 1 个值，即计算 y = Wx + b（模型函数）

    def forward(self, x):
        return self.linear(x)
        # 前向传播，返回线性层的输出，即模型的预测值y

# 生成一些简单的线性数据，用于测试模型
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 转换数据为PyTorch张量，用于模型输入
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 初始化两个模型，用于比较
model1 = LinearRegression()
model2 = LinearRegression()

# 确保两个模型的参数相同，用于比较
for param1, param2 in zip(model1.parameters(), model2.parameters()):
    param2.data.copy_(param1.data)

# 初始化自定义SGD优化器
sgd_custom = SGD(model1.parameters(), lr=0.01)

# 初始化PyTorch中的SGD优化器
sgd_torch = torch.optim.SGD(model2.parameters(), lr=0.01)

# 定义损失函数，均方误差损失函数
criterion = nn.MSELoss()

# 迭代次数
n_iterations = 1

# 训练模型，这个是自定义SGD优化器的训练过程
for iteration in range(n_iterations):
    # 自定义SGD优化器更新参数
    predictions_custom = model1(X_tensor)
    loss_custom = criterion(predictions_custom, y_tensor)
    loss_custom.backward()
    sgd_custom.step()
    sgd_custom.zero_grad()                        #清空上一轮的旧梯度

    # PyTorch中的SGD优化器更新参数
    predictions_torch = model2(X_tensor)
    loss_torch = criterion(predictions_torch, y_tensor)
    loss_torch.backward()
    sgd_torch.step()
    sgd_torch.zero_grad()

# 输出自定义SGD优化器得到的参数
for param in model1.parameters():
    print("自定义SGD优化器得到的参数:", param.data.item())

# 输出PyTorch中SGD优化器得到的参数
for param in model2.parameters():
    print("PyTorch中SGD优化器得到的参数:", param.data.item())

# 比较两个模型的参数，用已知的 y 去校正模型算出的 ŷ，已知的y是自定义SGD就散出来的标准答案，如果 y 本身就是模型函数，那损失永远是 0，没有任何可学的东西
