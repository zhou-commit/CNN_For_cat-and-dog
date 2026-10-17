import cv2

# 读取图像
image = cv2.imread("cat.7.jpg", cv2.IMREAD_GRAYSCALE)  # 请替换 "example.jpg" 为你要读取的图片路径

# 显示图像
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("灰度图像形状:", image.shape)
print("灰度图像像素值:")
print(image)

# 读取图像
rgb_image = cv2.imread("cat.7.jpg")  # 请替换 "example.jpg" 为你要读取的图片路径

# 打印RGB图像的形状和像素值
print("RGB图像形状:", rgb_image.shape)
print("RGB图像像素值:")
print(rgb_image)
