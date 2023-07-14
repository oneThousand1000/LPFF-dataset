import glob
import os
import shutil

import face_alignment
from skimage import io
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from tqdm import tqdm
import argparse
from ffhq_dataset.landmarks_detector import LandmarksDetector

def get_model(landmarks_model_path):

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
    landmarks_detector = LandmarksDetector(landmarks_model_path)

    return fa,landmarks_detector

if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    root = args.data_dir

    raw_image_dir = os.path.join(root, 'raw')
    output_dir = os.path.join(root, 'lm_facealignment_dlib')
    bad_image_dir = os.path.join(root, 'bad')
    shape_predictor_path = './checkpoints/shape_predictor_68_face_landmarks.dat'

    if not os.path.exists(shape_predictor_path):
        raise Exception(f'Please download shape predictor ckpt to {shape_predictor_path} according to readme.')

    os.makedirs(output_dir,exist_ok=True)
    os.makedirs(bad_image_dir,exist_ok=True)

    RAW_IMAGES_names = glob.glob(os.path.join(raw_image_dir,'*'))
    fa,landmarks_detector = get_model(landmarks_model_path = shape_predictor_path)

    pbar = tqdm(total = len(RAW_IMAGES_names))
    for index,raw_img_path in enumerate(RAW_IMAGES_names):
        pbar.update(1)

        name = os.path.basename(raw_img_path).split('.')[0]

        if len(glob.glob(os.path.join(output_dir,f'{name}_*.npy')))>0:
            continue

        count = 0
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            count += 1
            lm_path = os.path.join(output_dir,f'{name}_{i:02d}.npy')
            np.save(lm_path, face_landmarks)


        if count == 0:
            #print('landmarks_detector can not detect face, try to use face alignment')

            input_img = io.imread(raw_img_path)
            if len(input_img.shape)<3:
                continue
            origin_w, origin_h, _ = input_img.shape
            scale = 1.0
            if max(origin_w, origin_h) > 600:
                scale = 600 / max(origin_w, origin_h)
                input_img = cv2.resize(input_img, (int(origin_h * scale), int(origin_w * scale)))

            preds = fa.get_landmarks(input_img)

            if preds is not None:
                for face_landmarks in preds:
                    count += 1

                    # landmarks_by_dlib_and_face_alignment
                    lm_path = os.path.join(output_dir,f'{name}_{count:02d}.npy')
                    np.save(lm_path, face_landmarks/scale)
        if count==0:

            bad_path = os.path.join(bad_image_dir,f'{name}.png')
            #print('Detect no face, move to ', bad_path)
            shutil.move(raw_img_path,bad_path)


    