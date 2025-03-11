import cv2
import os
import numpy as np
import time
import shutil
from pathlib import Path
import sys
sys.path.append('./mav')
from mav import cal_mav




time = time.time


def enhance_mask(msk, window_size=1):
    msk_new = np.zeros_like(msk)
    for i in range(1, msk.shape[0] - window_size):
        for j in range(1, msk.shape[1] - window_size):
            if msk[i, j] == 0:
                continue
            elif msk[i - window_size, j] == 0 or msk[i, j - window_size] == 0 or msk[i + window_size, j] == 0 or msk[
                i, j + window_size] == 0 or msk[i + window_size, j + window_size] == 0 or msk[
                i - window_size, j - window_size] == 0:
                continue
            else:
                msk_new[i, j] = 1
    return msk_new


def generate_mask(img, value=(127, 127, 127)):
    pixels_to_change = np.all(img == value, axis=-1)
    img[pixels_to_change] = 0
    img[~pixels_to_change] = 1
    img = img[:, :, 0]
    return img


def load_ipt(img_path, mask_path, window_size=1):
    if img_path[-1] == '*':
        img_path = img_path[:-1]
        imgs = os.listdir(img_path)
        imgs = [img for img in imgs if (img.endswith('png') or img.endswith('jpg')) and img != 'mav_map.png']
        img_path = os.path.join(img_path, imgs[0])
    img = cv2.imread(img_path)
    img = ((img / 255 - 0.5) * 2).copy()
    if os.path.exists(mask_path):
        msk = (cv2.imread(mask_path) // 255)[:, :, 0]
    else:
        msk = generate_mask(img.copy(), (127, 127, 127))
    return img, enhance_mask(msk, window_size=window_size)



def select(mav_d, num, method='uniform', seed=271828):
    if seed is not None:
        seed = seed % 2 ** 32
        print(f"Current Random Seed: {seed}.\nIf you want to reproduce the results, please set the seed to 271828. If "
              f"you just want to generate an available result, you can set the seed to None or any number you like.\n"
              f"Anyway, this seed's impact is not significant.")
        np.random.seed(seed)
    mav_d = sorted(mav_d.items(), key=lambda x: x[1])
    mav = [dict(mav_d[len(mav_d) * i // 3:len(mav_d) * (i + 1) // 3]) for i in range(3)]
    res_file = []
    res_value = []
    num = num // 3
    for mav_d in mav:
        if method == 'uniform':
            mav_d = sorted(mav_d.items(), key=lambda x: x[1])
            filename = mav_d[::int(len(mav_d) / num + 0.51)]
            filename = [x[0] for x in filename]
            values = [x[1] for x in filename]
        elif method == 'max':
            mav_d = sorted(mav_d.items(), key=lambda x: x[1], reverse=True)
            filename = [x[0] for x in mav_d[:num]]
            values = [x[1] for x in filename]
        elif method == 'min':
            mav_d = sorted(mav_d.items(), key=lambda x: x[1])
            filename = [x[0] for x in mav_d[:num]]
            values = [x[1] for x in filename]
        elif method == 'normal' or method == 'gaussian':
            values = np.array(list(mav_d.values()))
            mean = np.mean(values)
            std = np.std(values)
            normal_samples = np.random.normal(mean, std, num)
            filename = []
            values_temp = []
            mav_d_copy = mav_d.copy()
            for sample in normal_samples:
                closest_key = min(mav_d_copy.keys(), key=lambda k: abs(mav_d_copy[k] - sample))
                mav_d_copy.pop(closest_key)
                filename.append(closest_key)
                values_temp.append(mav_d[closest_key])
            values = values_temp
        else:
            raise ValueError("Invalid method. Choose from 'uniform', 'max', 'min', or 'normal'/'gaussian'.")
        res_file += filename
        res_value += values
    return res_file, res_value


def organize_data(selected_files, method='delete', save_path=None, root_path=None, dns=True):
    counter = 0
    if dns is True:
        stage = [0, len(selected_files) // 3, len(selected_files) // 3 * 2]
        stage_name = ['Simple', 'Normal', 'Difficult']
    if method in ['copy', 'link'] and save_path is None:
        raise ValueError("Please specify the save path for 'copy' or 'link' method.")
    if method == 'delete':
        for file in os.listdir(root_path):
            if file not in selected_files:
                file_path = os.path.join(root_path, file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
    elif method == 'copy':
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        curr_stage = ''
        for file_path in selected_files:
            file = file_path[len(file_path) - file_path[::-1].index('/'):]
            root_path = file_path[:len(file_path) - file_path[::-1].index('/')]
            src_path = os.path.join(root_path, file)
            if dns:
                if counter in stage:
                    curr_stage = stage_name[stage.index(counter)]
                    stage.pop(0)
                    stage_name.pop(0)
                counter += 1
                if not os.path.exists(os.path.join(save_path, curr_stage)):
                    os.makedirs(os.path.join(save_path, curr_stage))
            dst_path = os.path.join(os.path.join(save_path, curr_stage), file)
            try:
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                elif os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
                print(f"Copied: {src_path} -> {dst_path}")
            except Exception as e:
                print(f"Failed to copy {src_path}: {e}")
    elif method == 'link':
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        curr_stage = ''
        for file_path in selected_files:
            file_path = file_path.replace('\\', '/')
            file = file_path[len(file_path) - file_path[::-1].index('/'):]
            root_path = file_path[:len(file_path) - file_path[::-1].index('/')]
            src_path = os.path.join(root_path, file)
            if dns:
                if counter in stage:
                    curr_stage = stage_name[stage.index(counter)]
                    stage.pop(0)
                    stage_name.pop(0)
                counter += 1
                if not os.path.exists(os.path.join(save_path, curr_stage)):
                    os.makedirs(os.path.join(save_path, curr_stage))
            dst_path = os.path.join(os.path.join(save_path, curr_stage), file)
            try:
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                src_path = Path(src_path)
                dst_path = Path(dst_path)
                src_absolute = src_path.resolve()
                dst_absolute = dst_path.resolve()
                dst_absolute.symlink_to(src_absolute)
                # os.symlink(src_path, dst_path)
                print(f"Created link: {src_path} -> {dst_path}")
            except Exception as e:
                print(f"Failed to create link {src_path}: {e}")


def main(data_path,save_path):
    """
    :param data_path: Can be a list of paths or a single path. All the datasets should be in the same format.
    :param save_path: The path to save organized data.
    :return: None
    """
    if type(data_path) is not list:
        data_path = [data_path]
    files = []
    for path in data_path:
        f = os.listdir(path)
        f = sorted(f)
        f = [os.path.join(path, x) for x in f]
        files += f
    counter = 0
    max_iter = -1
    mav_dict = {}
    print('--------Start Calculating MAV--------')
    for file in files:
        if counter != max_iter:
            counter += 1
            if counter % 100 == 0:
                print(f'%d complished' % counter)
        else:
            break
        if os.path.isfile(file):
            continue
        image = os.path.join(file, '*')
        mask = os.path.join(file, 'mask.png')
        img, msk = load_ipt(image, mask, window_size=1)
        mav, _ = cal_mav(img, msk, mode='4', method='mean', threshold=None)
        mav_dict[file] = mav
    print('--------Start Organizing Dataset--------')
    # print(mav_dict)
    selected, _ = select(mav_dict, 30000, method='normal')
    organize_data(selected, method='link', save_path=save_path,dns=True)


if __name__ == '__main__':
    main('../data','./DNS')
