# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./img/netease_bg.jpg', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# todo 模板透明背景需要处理
template = cv2.imread('./img/netease_template.png', cv2.IMREAD_IGNORE_ORIENTATION)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
h, w = template.shape[:2]
print('template:', h, w)
plt.imshow(img_gray)
plt.show()
plt.imshow(template)
plt.show()

res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
thresthold = 0.3
loc = np.where(res <= thresthold)
for pt in zip(*loc[::-1]):
    right_botton = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img, pt, right_botton, (0, 0, 255), 2)

plt.imshow(img)
plt.show()
