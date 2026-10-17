import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

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
# 加载训练好的猫狗分类模型（项目内 ./checkpoints/best.pt）
import os
ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "best.pt")
model = torch.load(ckpt_path, weights_only=False).to(device=device)
model.eval()

# 定义图像预处理（必须与训练时一致：先 Resize 再 ToTensor 再 Normalize）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 将图像数据转换为Tensor格式
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 标准化图像数据
])
# 类别标签（与 ImageFolder 自动编号一致：cat=0, dog=1）
class_labels = ["cat", "dog"]

def predict(image):
    """对上传的图片进行分类，并返回带分类结果的图片"""
    img = transform(image).unsqueeze(0)  # 预处理
    img =img.to(device)
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)  # 计算置信度（Softmax 归一化）
        confidence, predicted = torch.max(probabilities, 1)
        label = class_labels[predicted.item()]
        confidence_score = confidence.item() * 100  # 转换为百分比

    # 在图片上添加分类结果（上方居中）
    image_with_text = image.copy()
    draw = ImageDraw.Draw(image_with_text)

    # 设置字体（可替换为更大的字体，如 Arial）
    font = ImageFont.truetype("arial.ttf", 25)  # Windows 可能有 Arial.ttf

    # 计算文本位置（图片上方居中）
    text = f"{label} ({confidence_score:.2f}%)"
    bbox = draw.textbbox((0, 0), text, font=font)  # 获取文本边界框
    text_width = bbox[2] - bbox[0]
    image_width, _ = image_with_text.size
    text_x = (image_width - text_width) // 2
    text_y = 10  # 顶部 10px

    # 直接绘制文本
    draw.text((text_x, text_y), text, fill="red", font=font)

    return image, image_with_text

# 搭建 Gradio 界面
# 使用gradio，只需在原有的代码中增加几行，就能自动化生成交互式web页面，并支持多种输入输出格式，demo
# 统一图像框高度，使输出（分类结果）与输入图像大小一致
img_height = 400
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", height=img_height),
    outputs=[gr.Image(type="pil", label="原始图片", height=img_height), gr.Image(type="pil", label="分类结果", height=img_height)],
    title="猫狗图像分类系统",
    description="上传图片，左侧显示原始图片，右侧显示带分类结果的图片。",
)

# inbrowser=False：不自动打开系统外部浏览器，改用 IDE 内置预览打开
iface.launch(inbrowser=False)