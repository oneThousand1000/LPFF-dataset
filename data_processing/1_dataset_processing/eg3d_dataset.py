# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

#############################################################

'''

https://github.com/NVlabs/eg3d/blob/main/dataset_preprocessing/mirror_dataset.py

'''

#############################################################
import json
import numpy as np
from PIL import Image, ImageOps
import os
import glob
import argparse
COMPRESS_LEVEL = 0


def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--final_crop_dir", type=str)
    parser.add_argument("--camera_parameter_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    dest = args.output_dir
    final_crop_dir = args.final_crop_dir
    assert os.path.isdir(dest)

    # the original camera paramter json file of eg3d-FFHQ dataset
    dataset_file = os.path.join(dest, 'dataset-ffhq.json')

    if not os.path.exists(dataset_file):
        raise Exception(f'Please download camera parameter fils from https://drive.google.com/uc?id=14mzYD1DxUjh7BGgeWKgXtLHWwvr-he1Z and save it to {dataset_file}')

    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    new_dataset = dataset.copy()

    # read LPFF camera paramters
    with open(args.camera_parameter_path, "r") as f:
        lpff_camera_parameters = json.load(f)


    count = 139914
    image_list = glob.glob(os.path.join(final_crop_dir,'*.png'))
    for i, image_path in enumerate(image_list):
        ID = count + i * 2
        mirror_ID = count + i * 2 + 1


        dest_subdir = f'{int(ID // 1000):05d}'
        dest_mirror_subdir = f'{int(mirror_ID // 1000):05d}'
        dest_dir = os.path.join(dest, dest_subdir)
        dest_mirror_dir = os.path.join(dest, dest_mirror_subdir)
        os.makedirs(dest_dir, exist_ok=True)
        os.makedirs(dest_mirror_dir, exist_ok=True)

        image_save_path = os.path.join(dest_dir, f'img{ID:08d}.png')
        image_mirror_save_path = os.path.join(dest_mirror_dir, f'img{mirror_ID:08d}.png')

        fname = os.path.join(f'{dest_subdir}/img{ID:08d}.png')
        fname_mirror = os.path.join(f'{dest_mirror_subdir}/img{mirror_ID:08d}.png')

        print(f'Save image to {image_save_path}. Save mirror image to {image_mirror_save_path}')

        name = os.path.basename(image_path).split('.')[0]
        label = np.array(lpff_camera_parameters[name])
        assert len(label.shape) == 1 and label.shape[0] == 25

        img = Image.open(image_path)
        flipped_img = ImageOps.mirror(img)

        pose, intrinsics = np.array(label[:16]).reshape(4, 4), np.array(label[16:]).reshape(3, 3)
        flipped_pose = flip_yaw(pose)
        mirror_label = np.concatenate([flipped_pose.reshape(-1), intrinsics.reshape(-1)]).tolist()
        label = label.tolist()

        flipped_img.save(image_mirror_save_path, compress_level=COMPRESS_LEVEL)

        img.save(image_save_path, compress_level=COMPRESS_LEVEL)

        new_dataset["labels"].append([fname, label])
        new_dataset["labels"].append([fname_mirror, mirror_label])

    print('len(new_dataset): ', len(new_dataset["labels"]))
    data = json.dumps(new_dataset, indent=1)
    with open(os.path.join(dest, 'dataset.json'), "w", newline='\n') as f:
        f.write(data)

    os.remove(os.path.join(dest, 'dataset-ffhq.json'))
