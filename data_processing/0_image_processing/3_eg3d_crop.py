
import glob
import argparse
from PIL import ImageFile
import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d




def inverse_transformation(lm, lm_transformation):
    t = lm_transformation['t']
    w0 = lm_transformation['w0']
    h0 = lm_transformation['h0']
    s = lm_transformation['s']
    w = lm_transformation['w']
    target_size = lm_transformation['target_size']
    h = lm_transformation['h']

    lm = lm + np.reshape(
        np.array([(w / 2 - target_size / 2), (h / 2 - target_size / 2)]), [1, 2])

    lm /= s
    lm[:, 0] = lm[:, 0] + t[0] - w0 / 2
    lm[:, 1] = lm[:, 1] + t[1] - h0 / 2

    return lm




def crop_img(im_path, lm, lm3d_std, flip_lm=True):
    # to RGB
    im = Image.open(im_path).convert('RGB')
    W, H = im.size

    if flip_lm:
        lm[:, -1] = H - 1 - lm[:, -1]

    # lm_original = lm.copy()
    _, im, lm, _, lm_transformation = align_img(im, lm, lm3d_std, target_size=1024, rescale_factor=300)

    # debug = inverse_transformation(lm, lm_transformation)
    # print(debug - lm_original)

    return im




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


    lm3d_std = load_lm3d('./BFM')

    crop_dir = os.path.join(re_aligned_dir, "crop")
    os.makedirs(crop_dir, exist_ok=True)

    eg3d_crop_dir = os.path.join(re_aligned_dir, "crop/eg3d")
    os.makedirs(eg3d_crop_dir, exist_ok=True)

    stylegan_crop_dir = os.path.join(re_aligned_dir, "crop/stylegan")
    os.makedirs(stylegan_crop_dir, exist_ok=True)

    out_dir = os.path.join(re_aligned_dir, 'epoch_%s_%06d' % (20, 0))

    #eg3d_crop_check_dir= os.path.join(re_aligned_dir, "crop/eg3d-checked")

    seleted_dir=os.path.join(re_aligned_dir, 'debug')


    if not os.path.exists(out_dir):
        raise Exception('out_dir not exist! ')
    # =====================
    realign_size = 2400
    debug_img_list = glob.glob( os.path.join(seleted_dir, f'*.png'))
    pbar = tqdm.tqdm(total=len(debug_img_list))
    for debug_img_path in debug_img_list:
        pbar.update(1)
        
        img_name = os.path.basename(debug_img_path).split('.')[0]
        #checkedout_path = os.path.join(eg3d_crop_check_dir, f'{img_name}.png')
        # if os.path.exists(checkedout_path):
        #     continue

        out_path = os.path.join(eg3d_crop_dir, f'{img_name}.png')
        if os.path.exists(out_path):
            continue
        re_align_path = os.path.join(final_realign_path, f'{img_name}.png')


        pred_lm =np.load(os.path.join(final_5p_path,f'{img_name}.npy'))
        pred_lm[:, -1] = realign_size - 1 - pred_lm[:, -1]
        croped_img = crop_img(re_align_path, pred_lm, lm3d_std, flip_lm=False)
        left = int(croped_img.size[0] / 2 - 700 / 2)
        upper = int(croped_img.size[1] / 2 - 700 / 2)
        right = left + 700
        lower = upper + 700
        im_cropped = croped_img.crop((left, upper, right, lower))
        im_cropped = im_cropped.resize((512, 512), resample=Image.LANCZOS)

        im_cropped.save(out_path)


