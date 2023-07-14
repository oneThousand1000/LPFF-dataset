
import glob
import tqdm
import argparse
import cv2
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import numpy as np
from stylegan_image_align import stylegan_image_align



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    root = args.data_dir


    raw_image_dir = os.path.join(root, 'raw')
    re_aligned_dir = os.path.join(root, 'realign')
    detection_dir = os.path.join(re_aligned_dir, 'detections')

    assert os.path.isdir(re_aligned_dir) and os.path.isdir(detection_dir)

    debug_dir = os.path.join(re_aligned_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)

    final_realign_path = os.path.join(re_aligned_dir, 'realign')
    os.makedirs(final_realign_path, exist_ok=True)

    final_5p_path = os.path.join(re_aligned_dir, 'lm_5p_pred')
    os.makedirs(final_5p_path, exist_ok=True)

    # ============= deep face


    crop_dir = os.path.join(re_aligned_dir, "crop")
    os.makedirs(crop_dir, exist_ok=True)


    stylegan_crop_dir = os.path.join(re_aligned_dir, "crop/stylegan")
    os.makedirs(stylegan_crop_dir, exist_ok=True)

    realign_size = 2400
    out_dir = os.path.join(re_aligned_dir, 'epoch_%s_%06d' % (20, 0))

    eg3d_align_dir = os.path.join(re_aligned_dir, 'crop/eg3d')
    if not os.path.exists(out_dir):
        raise Exception('out_dir not exist! ')
    eg3d_align_img_list = glob.glob(os.path.join(eg3d_align_dir, f'*.png'))
    pbar = tqdm.tqdm(total=len(eg3d_align_img_list))
    for eg3d_align_img_path in eg3d_align_img_list:
        pbar.update(1)

        img_name = os.path.basename(eg3d_align_img_path).split('.')[0]

        re_align_path = os.path.join(final_realign_path, f'{img_name}.png')
        if os.path.exists( os.path.join(stylegan_crop_dir, f'{img_name}.png')):
            continue
        pred_lm = np.load(os.path.join(final_5p_path,f'{img_name}.npy'))
        stylegan_image_align(re_align_path, os.path.join(stylegan_crop_dir, f'{img_name}.png'),
                             pred_lm)