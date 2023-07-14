import argparse
import glob
from my_realign import realign_process_image

import os
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None




def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p


import tqdm
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    root = args.data_dir

    raw_image_dir = os.path.join(root,'raw')
    lm_facealignment_dlib_dir = os.path.join(root, 'lm_facealignment_dlib')

    re_aligned_dir = os.path.join(root,'realign')
    os.makedirs(re_aligned_dir, exist_ok=True)

    detection_dir = os.path.join(re_aligned_dir,'detections')
    os.makedirs(detection_dir, exist_ok=True)

    debug = False

    debug_dir = os.path.join(re_aligned_dir,'debug')
    os.makedirs(debug_dir, exist_ok=True)

    final_realign_path = os.path.join(re_aligned_dir,'realign')
    os.makedirs(final_realign_path, exist_ok=True)

    final_5p_path = os.path.join(re_aligned_dir,'lm_5p_pred')
    os.makedirs(final_5p_path, exist_ok=True)


    crop_dir = os.path.join(re_aligned_dir, "crop")
    os.makedirs(crop_dir, exist_ok=True)

    eg3d_crop_dir = os.path.join(re_aligned_dir, "crop/eg3d")
    os.makedirs(eg3d_crop_dir, exist_ok=True)

    stylegan_crop_dir = os.path.join(re_aligned_dir, "crop/stylegan")
    os.makedirs(stylegan_crop_dir, exist_ok=True)

    out_dir = os.path.join(re_aligned_dir, 'epoch_%s_%06d' % (20, 0))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # =====================
    realign_size = 2400
    # download
    landmarks_path_list = glob.glob(os.path.join(lm_facealignment_dlib_dir,'*.npy'))

    pbar = tqdm.tqdm(total=len(landmarks_path_list))
    for landmarks_path in landmarks_path_list:
        pbar.update(1)
        face_name = os.path.basename(landmarks_path).split('.')[0]
        raw_img_name = face_name[:-3]
        raw_img_path = glob.glob(os.path.join(raw_image_dir, f'{raw_img_name}.*'))[0]


        detection_path = os.path.join(detection_dir, f'{face_name}.txt')
        if os.path.exists(detection_path):
            continue
        if (not os.path.exists(raw_img_path)):
            print(f'raw img {raw_img_path} not exist!')
            continue

        face_landmarks = np.load(landmarks_path)

        re_align_path = os.path.join(re_aligned_dir, f'{face_name}.png')

        try:
            re_align_img, lm_new = realign_process_image(name=None, src_path=raw_img_path, dst_path=re_align_path,
                                                         output_size=realign_size, transform_size=4096,
                                                         enable_padding=True,
                                                         lm=face_landmarks)
        except:
            continue

        if lm_new is None:
            continue

        five_point_landmarks = extract_5p(lm_new)

        outLand = open(detection_path, "w")
        outLand.write(str(float(five_point_landmarks[0][0])) + " " + str(
            float(five_point_landmarks[0][1])) + "\n")
        outLand.write(str(float(five_point_landmarks[1][0])) + " " + str(
            float(five_point_landmarks[1][1])) + "\n")
        outLand.write(
            str(float(five_point_landmarks[2][0])) + " " + str(float(five_point_landmarks[2][1])) + "\n")
        outLand.write(str(float(five_point_landmarks[3][0])) + " " + str(
            float(five_point_landmarks[3][1])) + "\n")
        outLand.write(str(float(five_point_landmarks[4][0])) + " " + str(
            float(five_point_landmarks[4][1])) + "\n")
        outLand.close()

