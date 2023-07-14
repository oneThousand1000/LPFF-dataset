

import json
import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stylegan_dataset_dir", type=str, default=None)
    args = parser.parse_args()
    fnames = []
    stylegan_dataset_dir = args.stylegan_dataset_dir

    thetas_all = np.load('./files/camera_thetas.npy')
    phis_all = np.load('./files/camera_phis.npy')
    all_densitys = np.load('./files/density_FFHQ_LPFF.npy')

    for i in range(70000):
        fnames.append(f'{i // 1000:05d}/img{i:08d}.png')

    for i in range(139914, thetas_all.shape[0]):
        if i % 2 == 0:
            theta = thetas_all[i]
            phi = phis_all[i]

            density = all_densitys[i]
            duplicate_num = min(max(round(4 * 0.06 / density), 1), 4)
            # if density < 0.01:
            #     duplicate_num = 7
            # el
            if density < 0.02:
                duplicate_num = 6
            elif density < 0.03:
                duplicate_num = 5


            image_id = (i - 139914) // 2 + 70000
            for index in range(int(duplicate_num)):
                # fnames.append(f'{int(image_id // 1000 * 1000):05d}/{image_id:05d}.png')
                fnames.append(f'{image_id // 1000:05d}/img{image_id:08d}.png')
                # 00024/img00024642.png
                # 89586/img00089586.png
    print('\n',len(fnames),'samples!')
    data = json.dumps(fnames, indent=1)
    with open(os.path.join(stylegan_dataset_dir,f'FFHQ_LPFF_rebalanced.json'), "w", newline='\n') as f:
        f.write(data)

