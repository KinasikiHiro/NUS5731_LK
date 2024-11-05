import numpy as np
import cv2


def my_calcOpticalFlowPyrLK(prev_img, next_img, prev_pts, win_size=(10, 10), max_level=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)):
    # 将图像转换为灰度图
    # 检查输入图像是否为灰度图像
    if len(prev_img.shape) == 3:  # 如果是彩色图像
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_img
    if len(next_img.shape) == 3:  # 如果是彩色图像
        next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    else:
        next_gray = next_img


    # 初始化输出变量
    next_pts = np.zeros_like(prev_pts)
    status = np.zeros(prev_pts.shape[0], dtype=np.uint8)
    err = np.zeros(prev_pts.shape[0], dtype=np.float32)

    # 定义窗口大小
    half_win = win_size[0] // 2

    # 遍历每个特征点
    for i, pt in enumerate(prev_pts):
        x, y = pt[0]

        # 提取窗口区域
        Ix = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=5)
        It = next_gray - prev_gray

        A = np.zeros((2, 2))
        b = np.zeros((2, 1))

        for u in range(-half_win, half_win + 1):
            for v in range(-half_win, half_win + 1):
                if 0 <= x + u < prev_gray.shape[1] and 0 <= y + v < prev_gray.shape[0]:
                    Ix_val = Ix[int(y + v), int(x + u)]
                    Iy_val = Iy[int(y + v), int(x + u)]
                    It_val = It[int(y + v), int(x + u)]

                    A[0, 0] += Ix_val * Ix_val
                    A[0, 1] += Ix_val * Iy_val
                    A[1, 0] += Ix_val * Iy_val
                    A[1, 1] += Iy_val * Iy_val
                    b[0] += Ix_val * It_val
                    b[1] += Iy_val * It_val

        # 计算光流
        if np.linalg.det(A) != 0:
            nu = np.linalg.inv(A).dot(b)
            next_pts[i] = [[x + nu[0, 0], y + nu[1, 0]]]
            status[i] = 1
        else:
            next_pts[i] = [[x, y]]
            status[i] = 0

    return next_pts, status, err


cap = cv2.VideoCapture("Walkbright.mp4")
# cap = cv2.VideoCapture(0)

# Define parameters for corner detection
feature_params = dict(
    maxCorners = 10,  # Maximum number of corners
    qualityLevel = 0.01,  # Quality factor, used in corner detection. A higher value means higher quality corners, resulting in fewer selected corners
    minDistance = 20  # Used for Non-Maximum Suppression (NMS), suppresses all corners within a certain range of the strongest corner
)

"""
lk_params = dict(
    winSize=(10, 10),  # Size of the surrounding region to consider for nearby points
    maxLevel=2  # Maximum number of pyramid levels
)
"""
# Capture the first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, prev_img = cap.read()
prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

# Perform corner detection to get key points
prev_points = cv2.goodFeaturesToTrack(prev_img_gray, mask=None, **feature_params)

# Create a temporary canvas, so new drawings can be made on this mask and then added to the original image
mask_img = np.zeros_like(prev_img)

while True:

    ret, curr_img = cap.read()
    if curr_img is None:
        print("Video is over...")
        break
    curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    # Track points using optical flow
    curr_points, status, err = my_calcOpticalFlowPyrLK(prev_img_gray,curr_img_gray,prev_points,win_size = (5,5))
    """
    curr_points, status, err = cv2.calcOpticalFlowPyrLK(prev_img_gray,
                                                        curr_img_gray,
                                                        prev_points,
                                                        None,
                                                        **lk_params)
    """

    # print(status.shape)  # Values are either 1 or 0, with 1 indicating a successfully tracked point and 0 indicating tracking failure.
    good_new = curr_points[status == 1]
    good_old = prev_points[status == 1]

    # Draw on the image
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        mask_img = cv2.line(mask_img, pt1=(int(a), int(b)), pt2=(int(c), int(d)), color=(0, 0, 255), thickness=1)
        mask_img = cv2.circle(mask_img, center=(int(a), int(b)), radius=2, color=(255, 0, 0), thickness=2)

    # Overlay the canvas onto the original image and display it
    img = cv2.add(curr_img, mask_img)
    cv2.imshow("desct", img)
    if cv2.waitKey(60) & 0xFF == ord('q'):
        print("Quit...")
        break

    # Update the previous image and obtain new points
    prev_img_gray = curr_img_gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)
    if len(prev_points) < 5:
        # If there are too few matched points, re-detect corners on the current image
        prev_points = cv2.goodFeaturesToTrack(curr_img_gray, mask=None, **feature_params)
        mask_img = np.zeros_like(prev_img)  # Reset the canvas

cv2.destroyAllWindows()
cap.release()