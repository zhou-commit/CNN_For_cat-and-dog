# 猫狗图像分类（Cat vs Dog Image Classification）

基于 PyTorch 的猫狗二分类项目：使用预训练 ResNet-18 迁移学习完成训练，并通过 Gradio 提供图像交互推理界面。仓库同时包含一系列机器学习底层原理的学习脚本（手写 SGD、手写损失函数等）。

## 项目结构

```
demo1/
├── train.py              # 主训练脚本（ResNet-18 迁移学习 + 验证 + 测试）
├── gr_demo.py            # Gradio 图像分类交互界面（加载 checkpoints/best.pt）
├── dataload.py           # 数据加载演示（ImageFolder + DataLoader）
├── optimizer.py          # 学习脚本：手写 SGD 优化器，与 torch.optim.SGD 对比验证
├── loss.py               # 学习脚本：手写 MSE 损失函数，与 nn.MSELoss 对比验证
├── mlp.py                # 学习脚本：基础多层感知机（SimpleMLP）
├── simple_cnn.py         # 学习脚本：基础卷积网络（SimpleCNN）
├── pytorch_model.py      # 学习脚本：PyTorch 模型基础
├── image.py              # 学习脚本：OpenCV 读图与像素矩阵查看
├── 神经网络项目阅读指南.md  # 神经网络项目阅读方法论
└── cat-dog/              # 数据集目录（不随仓库分发，需自行放置，见下文）
```

## 环境要求

- Python 3.12（本项目使用 Miniconda 环境，请按自己的环境路径替换你的conda环境python路径）
- 主要依赖：`torch`（CUDA 版可选）、`torchvision`、`gradio`、`opencv-python`、`tqdm`、`numpy`

```powershell
conda create -n catdog python=3.12 -y
conda activate catdog
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install gradio opencv-python tqdm numpy
```

## 数据集准备

从阿里云天池下载猫狗数据集（cat-dog），解压后按以下结构放入项目根目录的 `cat-dog/` 文件夹：(注意文件是放在项目的根目录下，如果不在，需要再dataload中更改train.path)

```
cat-dog/
├── train/
│   ├── cat/    # 若干张 cat.xxx.jpg
│   └── dog/    # 若干张 dog.xxx.jpg
├── validate/
│   ├── cat/
│   └── dog/
└── test/
    ├── cat/
    └── dog/
```

标签由 `torchvision.datasets.ImageFolder` 按文件夹名自动生成（字母序：`cat=0, dog=1`），无需额外标注文件。

## 训练

```powershell
python train.py
```

训练流程要点（均可在 `train.py` 中修改）：

| 配置项 | 位置 | 默认值 |
|---|---|---|
| 模型 | 第 147 行 | 预训练 ResNet-18，最后一层替换为 2 类输出 |
| 优化器 / 学习率 | 第 183 行 | Adam，lr=0.001 |
| 损失函数 | 第 182 行 | CrossEntropyLoss |
| batch size | 第 170~175 行 | 64 |
| 训练轮数 epochs | 第 188 行 | 10 |

- 如果需要预训练，建议1先将 `SMOKE_TEST = False` 改为 `True` 做一次**冒烟测试**（仅取少量样本跑 2 个 epoch，几分钟验证全流程无报错），再改回 `False` 完整训练。
- 每个 epoch 结束会在验证集评估，`val_acc` 创新高时保存 `checkpoints/best.pt`，同时保存当轮快照 `cp{epoch}.pt`。
- 训练结束后自动加载 `best.pt` 在 `cat-dog/test` 上输出 `Test Accuracy`。

## 运行 Gradio 图像交互界面

训练产出 `checkpoints/best.pt` 后启动演示服务：

```powershell
python gr_demo.py
```

- 服务默认监听 `http://127.0.0.1:7860`，在浏览器打开该地址即可。
- 上传图片后，左侧显示原图，右侧显示带有类别标签和置信度的结果图（如 `cat (98.76%)`）。
- 若报 `torch.load` 相关错误，需确认加载时带有 `weights_only=False` 参数（PyTorch 2.6+ 的默认行为变化，本仓库代码已处理）。

## 注意事项

- 推理端 `gr_demo.py` 的图像预处理（Resize 224×224 + ImageNet 归一化）与训练端 `train.py` 严格一致，修改任一处必须同步另一处。
- 类别编号映射 `cat=0, dog=1` 由 `ImageFolder` 按文件夹字母序决定，更换数据集类别名时请同步更新 `gr_demo.py` 中的 `class_labels` 列表顺序。

## License

仅用于学习用途。
代码注释写的很详细了，如果作为初学者第一次接触神经网络并且想要好好阅读项目，可以细致阅读注释。
新人小白第一次在GitHub上进行项目开源部署，如果对你有帮助的话还请多多支持，点一个免费的star。
