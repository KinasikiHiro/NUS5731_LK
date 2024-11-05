import cv2
import numpy as np

def lucas_kanade_optical_flow(I1, I2, points, window_size=5):
    # 确保输入为灰度图像
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # 计算图像的梯度
    Ix = cv2.Sobel(I1, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(I1, cv2.CV_64F, 0, 1, ksize=5)
    It = I2.astype(np.float64) - I1.astype(np.float64)

    half_window = window_size // 2
    flow_vectors = []

    for point in points:
        x, y = point.ravel()
        if x < half_window or x >= I1.shape[1] - half_window or y < half_window or y >= I1.shape[0] - half_window:
            flow_vectors.append([np.nan, np.nan])  # 超出图像边界
            continue

        # 提取窗口内的图像梯度
        Ix_window = Ix[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].ravel()
        Iy_window = Iy[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].ravel()
        It_window = It[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].ravel()

        # 组装矩阵
        A = np.vstack((Ix_window, Iy_window)).T
        b = -It_window

        # 计算光流
        nu, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        flow_vectors.append(nu)

    return np.array(flow_vectors)




"""
w = window_size//2

    I1 = image1
    I1 = I1/255.0 # normalizing
    I2 = image2
    I2 = I2/255.0 # normalizing

    color = np.random.randint(0, 255, (100, 3))
    Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))  # for image 1 and image 2 in x direction
    Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))  # for image 1 and image 2 in y direction
    Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))  # for 1st image
    Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))  # for 2nd image

    Ix = (convolve2d(I1, Gx) + convolve2d(I2, Gx)) / 2 #smoothing in x direction

    Iy = (convolve2d(I1, Gy) + convolve2d(I2, Gy)) / 2 #smoothing in y direction
    It1 = convolve2d(I1, Gt1) + convolve2d(I2, Gt2)   #taking difference of two images using gaussian mask of all -1 and all 1

    I1 = np.float32(I1)
    I2 = np.float32(I2)
    features = prev_points

    feature = np.int32(features)

    feature = np.reshape(feature, newshape=[-1, 2])

    u = np.ones(Ix.shape)
    v = np.ones(Ix.shape)
    status=np.zeros(feature.shape[0]) # this will tell change in x,y
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    mask = np.zeros_like(image1)

    newFeature=np.zeros_like(feature)
    """Assumption is  that all the neighbouring pixels will have similar motion.
    Lucas-Kanade method takes a 3x3 patch around the point. So all the 9 points have the same motion.
    We can find (fx,fy,ft) for these 9 points. So now our problem becomes solving 9 equations with two unknown variables which is over-determined.
    A better solution is obtained with least square fit method.
    Below is the final solution which is two equation-two unknown problem and solve to get the solution.
                               U=Ainverse*B
    where U is matrix of 1 by 2 and contains change in x and y direction(x==U[0] and y==U[1])
    we first calculate A matrix which is 2 by 2 matrix of [[fx**2, fx*fy],[ fx*fy fy**2] and now take inverse of it
    and B is -[[fx*ft1],[fy,ft2]]"""

    for a,i in enumerate(feature):

        x, y = i


        A[0, 0] = np.sum((Ix[y - w:y + w+1, x - w:x + w + 1]) ** 2)

        A[1, 1] = np.sum((Iy[y - w:y + w+1, x - w:x + w + 1]) ** 2)
        A[0, 1] = np.sum(Ix[y - w:y + w+1, x - w:x + w + 1] * Iy[y - w:y + w+1, x - w:x + w + 1])
        A[1, 0] = np.sum(Ix[y - w:y + w+1, x - w:x + w + 1] * Iy[y - w:y + w + 1, x - w:x + w + 1])
        Ainv = np.linalg.pinv(A)

        B[0, 0] = -np.sum(Ix[y - w:y + w + 1, x - w:x + w + 1] * It1[y - w:y + w + 1, x - w:x + w + 1])
        B[1, 0] = -np.sum(Iy[y - w:y + w + 1, x - w:x + w + 1] * It1[y - w:y + w + 1, x - w:x + w + 1])
        prod = np.matmul(Ainv, B)

        u[y, x] = prod[0]
        v[y, x] = prod[1]

        newFeature[a]=[np.int32(x+u[y,x]),np.int32(y+v[y,x])]
        if np.int32(x+u[y,x])==x and np.int32(y+v[y,x])==y:    # this means that there is no change(x+dx==x,y+dy==y) so marking it as 0 else
            status[a]=0
        else:
            status[a]=1 # this tells us that x+dx , y+dy is not equal to x and y

    um=np.flipud(u)
    vm=np.flipud(v)

    good_new=newFeature[status==1] #status will tell the position where x and y are changed so for plotting getting only that points
    good_old = feature[status==1]

    return newFeature,status

"""
"""
# Convert images to float for more accurate calculations
    image1 = cv2.GaussianBlur(image1.astype(np.float32), (5, 5), 1)
    image2 = cv2.GaussianBlur(image2.astype(np.float32), (5, 5), 1)

    # Compute image gradients
    Ix = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x direction
    Iy = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y direction
    It = image2 - image1  # Temporal gradient between frames

    # Half window size
    w = window_size // 2

    # Prepare output arrays
    curr_points = []
    status = []

    # Loop through each point
    for pt in prev_points:
        x, y = pt.ravel().astype(int)

        # Ensure point is within bounds to apply the window
        if x - w < 0 or x + w >= image1.shape[1] or y - w < 0 or y + w >= image1.shape[0]:
            status.append(0)  # Mark as not trackable
            curr_points.append((0, 0))
            continue

        # Get image gradients in the window around the point
        Ix_win = Ix[y - w:y + w + 1, x - w:x + w + 1]
        Iy_win = Iy[y - w:y + w + 1, x - w:x + w + 1]
        It_win = It[y - w:y + w + 1, x - w:x + w + 1]

        # Construct matrices A and b for the equation Ax = b
        A = np.vstack((Ix_win.flatten(), Iy_win.flatten())).T
        b = -It_win.flatten()

        # Compute A^T A and A^T b
        ATA = A.T @ A
        ATb = A.T @ b

        # Regularization if ATA is close to singular
        if np.linalg.cond(ATA) > 1e10:  # Use condition number instead of determinant
            ATA += 1e-5 * np.eye(2)  # Regularization

        # Solve for flow (vx, vy) using least squares solution
        nu, _, _, _ = np.linalg.lstsq(ATA, ATb, rcond=None)  # Use lstsq for better numerical stability
        vx, vy = nu[0], nu[1]

        curr_points.append([x + vx, y + vy])
        status.append(1)  # Mark as successfully tracked

    # Convert results to expected format
    curr_points = np.array(curr_points, dtype=np.float32).reshape(-1, 1, 2)
    status = np.array(status, dtype=np.uint8)

    return curr_points, status
"""
