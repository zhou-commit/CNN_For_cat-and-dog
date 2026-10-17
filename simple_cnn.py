import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    #定义一个简单的卷积神经网络
    def __init__(self):
        super(SimpleCNN, self).__init__()
        #定义卷积层层，输入的通道数就是上一层的卷积核数量，其中每个卷积核的大小为3*3，padding用来固定卷积后的尺寸大小
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        #定义池化层,将尺寸直接减半，作用是只保留最强响应特征，减少计算量，加速训练
        self.pool = nn.MaxPool2d(2, 2)
        #定义全连接层，输入维度为64*28*28，输出维度为512
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)  # 输出维度（根据任务调整）比如你做猫狗分类那么就是2
        

    #定义前向传播过程
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        # 将图像特征图展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

my_model = SimpleCNN()
# 打印模型结构,train.py使用的就是该模型骨架，用卷积操作来代mlp的硬展平，提升分类的准确率