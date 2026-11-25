import torch,os
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

# 做对照实验，用来验证无CNN对模型的影响
if __name__ == '__main__':
    from train import train_loader,train_model,device,evaluate,val_loader
    torch.manual_seed(42)

    model = SimpleMLP().to(device)
    # 自己定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_model(model,train_loader,criterion,optimizer,val_loader=val_loader,num_epochs=10)

    # 训练出来的参数保存位置
    ckpt = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpoints','SimpleMLP','best.pt')
    best_model = SimpleMLP().to(device)
    best_model = torch.load(ckpt,map_location=device)
    val_acc,val_loss = evaluate(best_model,val_loader)
    print(f"val_acc:{val_acc:.4f}, val_loss:{val_loss:.4f}")

# mlp即多层感知器，nn.linear即全连接层
