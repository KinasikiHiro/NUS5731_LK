import cv2
import numpy as np

cap = cv2.VideoCapture("Walkbright.mp4")
# cap = cv2.VideoCapture(0)

# Define parameters for corner detection
feature_params = dict(
    maxCorners = 10,  # Maximum number of corners
    qualityLevel = 0.01,  # Quality factor, used in corner detection. A higher value means higher quality corners, resulting in fewer selected corners
    minDistance = 20  # Used for Non-Maximum Suppression (NMS), suppresses all corners within a certain range of the strongest corner
)

# Define parameters for Lucas-Kanade algorithm
lk_params = dict(
    winSize=(10, 10),  # Size of the surrounding region to consider for nearby points
    maxLevel=2  # Maximum number of pyramid levels
)

# Capture the first frame
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
    curr_points, status, err = cv2.calcOpticalFlowPyrLK(prev_img_gray,
                                                       curr_img_gray,
                                                       prev_points,
                                                       None,
                                                       **lk_params)
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
        print("Bye...")
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