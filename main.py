import cv2
import numpy as np

# 读取图片
image = cv2.imread('./sources/binary_img.png')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行二值化
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 创建一个与原图大小相同的全黑图像
red_image = np.zeros_like(image)

# 将白色区域变为红色
red_image[binary == 255] = [0, 0, 255]  # BGR格式，红色为[0, 0, 255]

# 显示结果
cv2.imshow('Red Image', red_image)
cv2.imwrite('./outputs_without_adaptive/Segmentation.png', red_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
