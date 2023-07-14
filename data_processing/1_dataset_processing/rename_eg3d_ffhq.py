import gdown
import json
import numpy as np
from PIL import Image, ImageOps
import os
from tqdm import tqdm
import argparse
import glob
import argparse
COMPRESS_LEVEL = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    dest = args.output_dir
    assert os.path.isdir(dest)

    dataset_file = os.path.join(dest, 'dataset-ffhq.json')

    if not os.path.exists(dataset_file):
        raise Exception(f'Please download camera parameter fils from https://drive.google.com/uc?id=14mzYD1DxUjh7BGgeWKgXtLHWwvr-he1Z and save it to {dataset_file}')

    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    new_dataset = {"labels":[]}
    for item in dataset["labels"]:
        name = item[0]
        cp_data = item[1]
        ID = int( name[:5])

        new_name = f'{int(ID*2 // 1000):05d}/img{ID*2:08d}.png' if 'mirror' not in  name else f'{int((ID*2+1) // 1000):05d}/img{ID*2+1:08d}.png'

        new_dataset["labels"].append([new_name, cp_data])

    os.remove(dataset_file)

    for image_path in glob.glob(os.path.join(dest, '*/*')):
        name = os.path.basename(image_path)
        ID =int( name[:5])
        new_name = f'img{ID * 2:08d}.png' if 'mirror' not in name else f'img{ID * 2 + 1:08d}.png'
        os.rename(image_path,image_path.replace(name,new_name))

    data = json.dumps(new_dataset, indent=1)
    with open(os.path.join(dest, 'dataset-ffhq.json'), "w", newline='\n') as f:
        f.write(data)