import numpy as np


def cal_mat_angle(img1, img2):
    dot = np.sum(img1 * img2, axis=2)
    norm = np.linalg.norm(img1, axis=2) * np.linalg.norm(img2, axis=2)
    cos = dot / (norm + 1e-8)
    cos = np.clip(cos, -1, 1)
    angle = np.arccos(cos)
    return np.rad2deg(angle)


def cal_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (norm_v1 * norm_v2)
    angle_in_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees


def cal_mav(img, msk, mode=3, method='mean', threshold=None):
    """
    :param img: Target Surface Normal
    :param msk: Mask, you can generate it by using generate_mask()
    :param mode: MAV's windowsize, String '4' means 4-neighbor, Int 4 means 4*4 window.
    :param method: How to calculate MAV in a window, 'mean' means average, 'min' means minimum, 'max' means maximum.
    :param threshold: If you want to calculate the percentage of MAV greater than a certain value, set the threshold.
    :return: Result and MAV map
    """
    mav = np.zeros_like(msk, dtype=np.float32)
    if mode == '4':
        img_left = img[:, 1:, :]
        img_down = img[1:, :, :]
        mav_h = cal_mat_angle(img_left, img[:, :-1, :])
        mav_v = cal_mat_angle(img_down, img[:-1, :, :])
        mav_h = np.append(mav_h, np.zeros((img.shape[0], 1)), axis=1)
        mav_v = np.append(mav_v, np.zeros((1, img.shape[1])), axis=0)
        if method == 'mean':
            mav = (mav_h + mav_v) / 2
        elif method == 'min':
            mav = np.minimum(mav_v, mav_h)
        elif method == 'max':
            mav = np.maximum(mav_v, mav_h)
        mav *= msk
    elif type(mode) is int:
        img_crop = img[mode // 2:-(mode // 2), mode // 2:-(mode // 2), :].copy()
        mav = np.zeros_like(msk, dtype=np.float32)
        for m in range(-(mode // 2), mode // 2 + 1):
            for n in range(-(mode // 2), mode // 2 + 1):
                if m == 0 and n == 0:
                    continue
                if -(mode // 2) + m == 0 and -(mode // 2) + n == 0:
                    img_shift = img[mode // 2 + m:, mode // 2 + n:, :].copy()
                elif -(mode // 2) + m == 0:
                    img_shift = img[mode // 2 + m:, mode // 2 + n:-(mode // 2) + n, :].copy()
                elif -(mode // 2) + n == 0:
                    img_shift = img[mode // 2 + m:-(mode // 2) + m, mode // 2 + n:, :].copy()
                else:
                    img_shift = img[mode // 2 + m:-(mode // 2) + m, mode // 2 + n:-(mode // 2) + n, :].copy()
                res = np.pad(cal_mat_angle(img_crop, img_shift), ((mode // 2, mode // 2), (mode // 2, mode // 2)),
                             mode='constant', constant_values=0)
                if method == 'max':
                    mav = np.maximum(mav, res)
                elif method == 'min':
                    mav = np.minimum(mav, res)
                else:
                    mav += res
        if method == 'mean':
            mav = mav / (mode * mode - 1)
    if threshold is not None:
        result = np.sum(mav > threshold) / np.sum(msk) * 100
    else:
        result = np.sum(mav) / np.sum(msk)
    return result, mav * msk
