# coding=utf-8
"""滑块验证码缺口位置定位"""
"""
对于netease_bg.jpg滑块边缘颜色区分度不明显，边缘检测效果较差，
无法形成封闭形状，此方法就行不通
"""
import cv2
import matplotlib.pyplot as plt

# 1.读取图片
img = cv2.imread('./img/tencent_bg.jpg')
plt.imshow(img)
plt.show()
result = img.copy()
# 2.高斯模糊处理
blurred = cv2.GaussianBlur(img, (5, 5), 0)
plt.imshow(blurred)
plt.show()

# 3.Canny边缘检测
canny = cv2.Canny(blurred, 200, 400)
plt.imshow(canny)
plt.show()
# 4.轮廓检测
contours, hierarchy = cv2.findContours(canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):  # 所有轮廓
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
plt.imshow(img)
plt.show()
# 5.获取位置
# 对轮廓的面积或周长范围做限制，就能过滤出目标轮廓的位置，前提是我们对目标位置的轮廓大小是预先确定的
for i, contour in enumerate(contours):  # 所有轮廓
    if 7000 < cv2.contourArea(contour) <= 9000 and 300 < cv2.arcLength(contour, True) < 500:
        # 轮廓面积大概7000到9000之间，周长在300到500之间
        x, y, w, h = cv2.boundingRect(contour)  # 外接矩形
        print(x, y, w, h)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite('img/tencent_bg_detect_output.jpg', result)
        plt.imshow(result)
        plt.title('detect output')
        plt.show()
        print('缺口位置：', x)
        break
