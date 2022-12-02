import numpy as np
import matplotlib.pyplot as plt
from Models.base_solver import get_ground_truth_3D_occ, get_ground_truth_3D_ori, get_conditional_input_data
import cv2


def draw_input_info(fId, x):
    # x = > T * H * W * C
    window_size = x.shape[0]
    fig = plt.figure(fId, figsize=(10 * window_size, 30))

    for i in range(window_size):
        depData, oriData,_ = np.split(x[i], [1, 3], axis=-1)
        depImg = depData.squeeze()
        oriImg = np.concatenate([oriData, np.zeros(shape=[oriData.shape[0], oriData.shape[1], 1], dtype=oriData.dtype)], axis=-1)

        y = fig.add_subplot(2, window_size, i + 1)
        y.imshow(depImg, cmap='gray')
        y = fig.add_subplot(2, window_size, window_size + i + 1)
        y.imshow(oriImg)


def draw_vox_slice(fId, V, sliceID):
    fig = plt.figure(fId, figsize=(10, 10))
    sliceImg = V[sliceID, :, :, :].copy()
    mask = (sliceImg ** 2).sum(-1) > 1e-3
    sliceImg[mask, :] = (sliceImg[mask, :] + 1.0) * 0.5
    sliceImg = np.clip(sliceImg, 0, 1)
    y = fig.add_subplot(1, 1, 1)
    y.imshow(sliceImg)
    return fig


# has bug
def draw_vox_total(fId, V, dd = 1):
    flag = False
    maskA = None
    Img = np.zeros(shape=[V.shape[1], V.shape[2], 3], dtype=np.float32)

    for sliceID in range(V.shape[0] // dd):
        sliceImg = V[sliceID, :, :, :]
        maskB = (sliceImg ** 2).sum(-1) > 1e-3  # H * W
        if (not flag):
            flag = True
            maskA = maskB.copy()
            Img[maskB, :] = (sliceImg[maskB, :] + 1.0) * 0.5
        else:
            # voxels to be updated = current seen - previous seen
            mask = np.logical_xor(np.logical_or(maskA, maskB), maskA)
            Img[mask, :] = (sliceImg[mask, :] + 1.0) * 0.5
            maskA = np.logical_or(maskA, maskB)

    fig = plt.figure(fId, figsize=(10, 10))
    y = fig.add_subplot(1, 1, 1)
    y.imshow(Img)


def draw_weights_slice(fId, V, sliceID):
    fig = plt.figure(fId, figsize=(10, 10))
    sliceImg = V[sliceID, :, :, :].copy().squeeze()
    y = fig.add_subplot(1, 1, 1)
    y.imshow(sliceImg)


def draw_weights_total(fId, V, W):
    flag = False
    maskA = None
    Img = np.zeros(shape=[V.shape[1], V.shape[2], 1], dtype=np.float32)

    for sliceID in range(V.shape[0]):
        sliceImg = V[sliceID, :, :, :]
        weightImg = W[sliceID, :, :, :]

        maskB = (sliceImg ** 2).sum(-1) > 1e-3  # H * W
        if (not flag):
            flag = True
            maskA = maskB.copy()
            Img[maskB, :] = weightImg[maskB, :]
        else:
            mask = np.logical_xor(np.logical_or(maskA, maskB), maskA)
            Img[mask, :] = weightImg[mask, :]
            maskA = np.logical_or(maskA, maskB)

    fig = plt.figure(fId, figsize=(10, 10))
    y = fig.add_subplot(1, 1, 1)
    y.imshow(Img.squeeze())


def get_vox_total_pic(V, dd=1):
    flag = False
    maskA = None
    Img = np.zeros(shape=[V.shape[1], V.shape[2], 3], dtype=np.float32)

    for sliceID in range(V.shape[0] // dd):
        sliceImg = V[sliceID, :, :, :]
        maskB = (sliceImg ** 2).sum(-1) > 1e-3  # H * W
        if (not flag):
            flag = True
            maskA = maskB.copy()
            Img[maskB, :] = (sliceImg[maskB, :] + 1.0) * 0.5
        else:
            mask = np.logical_xor(np.logical_or(maskA, maskB), maskA)
            Img[mask, :] = (sliceImg[mask, :] + 1.0) * 0.5
            maskA = np.logical_or(maskA, maskB)

    return Img * 255


def get_vox_slice_pic(V, sliceID=48):
    sliceImg = V[sliceID, :, :, :].copy()
    mask = (sliceImg ** 2).sum(-1) > 1e-3
    sliceImg[mask, :] = (sliceImg[mask, :] + 1.0) * 0.5
    sliceImg = np.clip(sliceImg, 0, 1)
    return sliceImg*255


def draw_arrows_by_projection(fileDir):

    h = 128
    w = 128
    d = 96
    flip = True
    noise = True
    target = get_conditional_input_data(fileDir, flip, noise, image_size=1024) * 255
    hair_ori = get_ground_truth_3D_ori(fileDir, flip)

    image = get_vox_total_pic(hair_ori) / 255.
    mask = (image**2).sum(-1) > 0
    image = image * 2 - 1

    for hh in range(h):
        for ww in range(w):
            if mask[hh, ww]:

                o = image[hh, ww][:2]
                o /= np.sqrt(np.sum(o**2) + 1e-8)
                o[1] *= -1

                # radius = 8
                o *= 4
                center = np.array([ww * 8 + 4, hh * 8 + 4])
                pt1 = (center - o).astype(np.int32)
                pt2 = (center + o).astype(np.int32)

                cv2.arrowedLine(target, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 1)

    cv2.imwrite("sss.jpg", target)


# draw_arrows_by_projection("E:\wukeyu\hair_reconstruction\TrainData/video1/frame3")