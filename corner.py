import cv2
import numpy as np

feature_params = dict(
    maxCorners = 100,  # Maximum number of corners
    qualityLevel = 0.01,  # Quality factor, used in corner detection. A higher value means higher quality corners, resulting in fewer selected corners
    minDistance = 30  # Used for Non-Maximum Suppression (NMS), suppresses all corners within a certain range of the strongest corner
)
cap = cv2.VideoCapture("Walkbright.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, prev_img = cap.read()
prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

# Perform corner detection to get key points
prev_points = cv2.goodFeaturesToTrack(prev_img_gray, mask=None, **feature_params)

# 绘制找到的特征点
for point in prev_points:
    x, y = point.ravel()  # 解包点的坐标
    cv2.circle(prev_img, (int(x), int(y)), 5, (255, 0, 0), -1)  # 绘制圆圈

# 显示带有特征点的图像
cv2.imshow("Features", prev_img)
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()  # 关闭窗口
cap.release()  # 释放视频捕获对象