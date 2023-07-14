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
import os
import argparse
import glob

import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_crop_dir", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    dest = args.output_dir
    final_crop_dir = args.final_crop_dir
    assert os.path.isdir(dest)


    count = 70000  
    image_list = glob.glob(os.path.join(final_crop_dir,'*.png'))
    for i, image_path in enumerate(image_list):
        ID = count + i

        dest_subdir = f'{int(ID // 1000 * 1000):05d}'
        dest_dir = os.path.join(dest, dest_subdir)
        os.makedirs(dest_dir, exist_ok=True)

        image_save_path = os.path.join(dest_dir, f'{ID:05d}.png')
        fname = os.path.join(f'{dest_subdir}\\{ID:05d}.png')

        print(f'save image to {image_save_path}')
        shutil.copy(image_path,image_save_path)

