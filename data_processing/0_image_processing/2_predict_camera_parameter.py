
import glob
import shutil
import cv2
import argparse
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from options.test_options import TestOptions
from models import create_model
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch
from util.visualizer import MyVisualizer
import pickle


def compute_rotation(angles):
    """
    Return:
        rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

    Parameters:
        angles           -- torch.tensor, size (B, 3), radian
    """

    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1])
    zeros = torch.zeros([batch_size, 1])
    x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)[0]
def convert_cart_to_spherical(coords):
    '''
    Vectorized converter from cartesian coordinates to spherical coordinates
    :param coords:
    :return: radius, thetas
    '''

    assert coords.shape[1] >= 2

    def _single_angle(current, rest):
        radii = np.linalg.norm(rest, axis=1)
        theta = np.arctan2(radii, current)
        return theta

    radii = np.linalg.norm(coords, axis=1)
    num_thetas = coords.shape[1] - 1 # one fewer parameters since radius is a parameter
    thetas = np.zeros(shape=(coords.shape[0], num_thetas))

    for i in range(num_thetas):
        current = coords[:,i]
        rest = coords[:,i+1:]
        theta = _single_angle(current, rest)
        if i == (num_thetas - 1):
            ids = np.argwhere(coords[:,-1] < 0)
            if len(ids) > 0:
                theta[ids] = 2*np.pi - theta[ids]

        thetas[:,i] = theta
    '''
        flip to align with LookAtPoseSampler.sample.

        thetas[0] = [theta,phi]

        Usage: 
        LookAtPoseSampler.sample(theta, phi, cam_pivot, radius=cam_radius, device=device)

        '''
    thetas[0, 0] = np.pi - thetas[0, 0]
    return radii, thetas
def compute_pose(coeff):

    angle = coeff['angle']
    trans = coeff['trans'][0]
    #print('angle:',angle)
    R = compute_rotation(torch.from_numpy(angle)).numpy()

    trans[2] += -10
    c = -np.dot(R, trans)
    pose = np.eye(4)
    pose[:3, :3] = R

    c *= 0.27  # factor to match tripleganger

    c[1] += 0.006  # offset to align to tripleganger
    c[2] += 0.161  # offset to align to tripleganger
    c = c / np.linalg.norm(c) * 2.7

    pose[0, 3] = c[0]
    pose[1, 3] = c[1]
    pose[2, 3] = c[2]

    Rot = np.eye(3)
    Rot[0, 0] = 1
    Rot[1, 1] = -1
    Rot[2, 2] = -1
    pose[:3, :3] = np.dot(pose[:3, :3], Rot)

    return pose

def load_kernel():
    with open('./kde.pkl', 'rb') as f:  # open file with write-mode
        kernel = pickle.load(f)
    return kernel

def compute_density(kernel,pose):
    co = pose[:, 3:4]

    radius, thetas = convert_cart_to_spherical(np.reshape(co[:3, :], (1, 3)))
    theta = thetas[0][0]
    phi = thetas[0][1]
    density = kernel(np.array([theta, phi]))
    return density,theta,phi




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


def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB
    im = Image.open(im_path).convert('RGB')
    W, H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]

    # lm_original = lm.copy()
    _, im, lm, _, lm_transformation = align_img(im, lm, lm3d_std)

    # debug = inverse_transformation(lm, lm_transformation)
    # print(debug - lm_original)

    if to_tensor:
        im = torch.tensor(np.array(im) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm, lm_transformation




def compute_rotation(angles):
    """
    Return:
        rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

    Parameters:
        angles           -- torch.tensor, size (B, 3), radian
    """

    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1])
    zeros = torch.zeros([batch_size, 1])
    x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)[0]

import tqdm
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    root = args.data_dir


    raw_image_dir = os.path.join(root, 'raw')
    re_aligned_dir = os.path.join(root, 'realign')
    detection_dir = os.path.join(re_aligned_dir, 'detections')

    # the density of FFHQ
    kde_kernel = load_kernel()

    bad_re_align_dir = os.path.join(re_aligned_dir, 'bad_re_align')
    os.makedirs(bad_re_align_dir, exist_ok=True)

    debug_dir = os.path.join(re_aligned_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)

    final_realign_path = os.path.join(re_aligned_dir, 'realign')
    os.makedirs(final_realign_path, exist_ok=True)

    final_5p_path = os.path.join(re_aligned_dir, 'lm_5p_pred')
    os.makedirs(final_5p_path, exist_ok=True)



    # ============= Deep3DFaceRecon_pytorch model =================
    opt = TestOptions().parse()
    opt.name = 'pretrained'
    opt.img_folder = re_aligned_dir
    opt.epoch = 20
    visualizer = MyVisualizer(opt)
    device = torch.device(0)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    lm3d_std = load_lm3d(opt.bfm_folder)
    # ============================================================


    out_dir = os.path.join(re_aligned_dir, 'epoch_%s_%06d' % (opt.epoch, 0))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # =====================
    realign_size = 2400
    re_align_img_list = glob.glob( os.path.join(re_aligned_dir, f'*.png'))
    pbar = tqdm.tqdm(total=len(re_align_img_list))
    for idx,re_align_path in enumerate(re_align_img_list):
        pbar.update(1)
        
        img_name = os.path.basename(re_align_path).split('.')[0]

        bad_re_align_path = os.path.join(bad_re_align_dir,os.path.basename(re_align_path))
        detection_path = os.path.join(detection_dir, f'{img_name}.txt')
        if os.path.exists(os.path.join(final_5p_path,f'{img_name}.npy')) or not os.path.exists(detection_path):
            continue


        rescale_factor = 466.285
        # im, lm,lm_transformation
        im_tensor, lm_tensor, lm_transformation = read_data(re_align_path, detection_path, lm3d_std)
        #
        data = {
            'imgs': im_tensor,
            'lms': lm_tensor
        }
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference


        coeff = model.get_coeff()  # save predicted coefficients

        pose = compute_pose(coeff)

        density, theta, phi = compute_density(kde_kernel, pose)
 

        if density >= 0.4:
            # remove the religned image with density > 0.4 in FFHQ distribution
            shutil.move(re_align_path, bad_re_align_path)
            continue



        model.save_coeff(
                os.path.join(out_dir, f'{img_name}' + '.mat'))  # save predicted coefficients
        visuals = model.get_current_visuals()  # get image results
        visualizer.display_current_results(visuals, 0, opt.epoch, dataset='flickr',
                                               save_results=True, save_path=debug_dir,count=0, name=img_name, add_image=False)

        pred_lm = model.pred_lm[0].detach().cpu().numpy()
        pred_lm = inverse_transformation(pred_lm, lm_transformation)


        pred_lm[:, -1] = realign_size - 1 - pred_lm[:, -1]
        np.save(os.path.join(final_5p_path,f'{img_name}.npy'),pred_lm)
       
        shutil.move(re_align_path,os.path.join(final_realign_path,f'{img_name}.png'))

