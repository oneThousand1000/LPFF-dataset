

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

    for i in range(89590):
        # fnames.append(f'{int(i// 1000*1000):05d}/{i:05d}.png')
        fnames.append(f'{i // 1000:05d}/img{i:08d}.png')




    print('\n',len(fnames),'samples!')
    data = json.dumps(fnames, indent=1)
    with open(os.path.join(stylegan_dataset_dir,f'FFHQ_LPFF.json'), "w", newline='\n') as f:
        f.write(data)

