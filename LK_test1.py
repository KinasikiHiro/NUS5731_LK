import cv2
import numpy as np
from scipy.signal import convolve2d

def my_calcOpticalFlowPyrLK(image1, image2, prev_points, window_size=5):
    """
    Custom implementation of the Lucas-Kanade optical flow algorithm.

    Parameters:
    - image1: First grayscale image (previous frame).
    - image2: Second grayscale image (next frame).
    - prev_points: Keypoints in the previous frame to track.
    - window_size: Size of the window around each point to consider for calculations (must be odd).

    Returns:
    - curr_points: Calculated new positions of keypoints in the next frame.
    - status: Array indicating whether each point was successfully tracked.
    """
    w = window_size // 2

    I1 = image1
    I1 = I1 / 255.0  # normalizing
    I2 = image2
    I2 = I2 / 255.0  # normalizing

    color = np.random.randint(0, 255, (100, 3))
    Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))  # for image 1 and image 2 in x direction
    Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))  # for image 1 and image 2 in y direction
    Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))  # for 1st image
    Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))  # for 2nd image

    Ix = (convolve2d(I1, Gx) + convolve2d(I2, Gx)) / 2  # smoothing in x direction

    Iy = (convolve2d(I1, Gy) + convolve2d(I2, Gy)) / 2  # smoothing in y direction
    It1 = convolve2d(I1, Gt1) + convolve2d(I2,
                                           Gt2)  # taking difference of two images using gaussian mask of all -1 and all 1

    I1 = np.float32(I1)
    I2 = np.float32(I2)
    features = prev_points

    feature = np.int32(features)

    feature = np.reshape(feature, newshape=[-1, 2])

    u = np.ones(Ix.shape)
    v = np.ones(Ix.shape)
    status = np.zeros(feature.shape[0])  # this will tell change in x,y
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    mask = np.zeros_like(image1)

    newFeature = np.zeros_like(feature)
    """
    Assumption is  that all the neighbouring pixels will have similar motion.
    Lucas-Kanade method takes a 3x3 patch around the point. So all the 9 points have the same motion.
    We can find (fx,fy,ft) for these 9 points. So now our problem becomes solving 9 equations with two unknown variables which is over-determined.
    A better solution is obtained with least square fit method.
    Below is the final solution which is two equation-two unknown problem and solve to get the solution.
                               U=Ainverse*B
    where U is matrix of 1 by 2 and contains change in x and y direction(x==U[0] and y==U[1])
    we first calculate A matrix which is 2 by 2 matrix of [[fx**2, fx*fy],[ fx*fy fy**2] and now take inverse of it
    and B is -[[fx*ft1],[fy,ft2]]
    """

    for a, i in enumerate(feature):

        x, y = i

        A[0, 0] = np.sum((Ix[y - w:y + w + 1, x - w:x + w + 1]) ** 2)

        A[1, 1] = np.sum((Iy[y - w:y + w + 1, x - w:x + w + 1]) ** 2)
        A[0, 1] = np.sum(Ix[y - w:y + w + 1, x - w:x + w + 1] * Iy[y - w:y + w + 1, x - w:x + w + 1])
        A[1, 0] = np.sum(Ix[y - w:y + w + 1, x - w:x + w + 1] * Iy[y - w:y + w + 1, x - w:x + w + 1])
        Ainv = np.linalg.pinv(A)

        B[0, 0] = -np.sum(Ix[y - w:y + w + 1, x - w:x + w + 1] * It1[y - w:y + w + 1, x - w:x + w + 1])
        B[1, 0] = -np.sum(Iy[y - w:y + w + 1, x - w:x + w + 1] * It1[y - w:y + w + 1, x - w:x + w + 1])
        prod = np.matmul(Ainv, B)

        u[y, x] = prod[0]
        v[y, x] = prod[1]

        newFeature[a] = [np.int32(x + u[y, x]), np.int32(y + v[y, x])]
        if np.int32(x + u[y, x]) == x and np.int32(
                y + v[y, x]) == y:  # this means that there is no change(x+dx==x,y+dy==y) so marking it as 0 else
            status[a] = 0
        else:
            status[a] = 1  # this tells us that x+dx , y+dy is not equal to x and y

    um = np.flipud(u)
    vm = np.flipud(v)

    good_new = newFeature[
        status == 1]  # status will tell the position where x and y are changed so for plotting getting only that points
    good_old = feature[status == 1]

    return newFeature, status

cap = cv2.VideoCapture("Walkbright.mp4")
# cap = cv2.VideoCapture(0)

# Define parameters for corner detection
feature_params = dict(
    maxCorners = 100,  # Maximum number of corners
    qualityLevel = 0.01,  # Quality factor, used in corner detection. A higher value means higher quality corners, resulting in fewer selected corners
    minDistance = 30  # Used for Non-Maximum Suppression (NMS), suppresses all corners within a certain range of the strongest corner
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
    curr_points, status = my_calcOpticalFlowPyrLK(prev_img_gray,curr_img_gray,prev_points,window_size = 5)
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