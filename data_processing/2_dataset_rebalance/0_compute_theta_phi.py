import json
import math
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from tqdm import tqdm
from camera_utils import LookAtPoseSampler


def convert_cart_to_spherical(coords):
    '''
    Vectorized converter from cartesian coordinates to spherical coordinates
    :param coords:
    :return: radius, thetas
    '''
    #print(coords.shape)
    x, y, z = coords[0]
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)

    phi = math.acos(y / r)  # 0 - pi
    theta = np.pi - math.atan2(z, x)  #

    # v = torch.clamp(v, 1e-5, math.pi - 1e-5)
    # theta = h
    # v = v / math.pi
    # phi = torch.arccos(1 - 2 * v)

    h = theta
    # 1- 2v = cos(phi)

    v = (1 - math.cos(phi)) / 2 * math.pi


    '''
    
    coords: 
    h \in [0,2pi]
    v \in [0,pi]
    
    v = torch.clamp(v, 1e-5, math.pi - 1e-5)
    theta = h
    v = v / math.pi
    phi = torch.arccos(1 - 2 * v)

    camera_origins = torch.zeros((batch_size, 3), device=device)
    camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
    camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
    camera_origins[:, 1:2] = radius*torch.cos(phi)
    
    
    '''

    return h, v


def convert_spherical_to_cart(h,v,radii):
    theta = h
    v = v / math.pi
    phi = np.arccos(1 - 2 * v)

    x = radii * np.sin(phi) * np.cos(math.pi - theta)
    z = radii * np.sin(phi) * np.sin(math.pi - theta)
    y = radii * np.cos(phi)
    return np.array([x,y,z])



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--eg3d_dataset_dir", type=str, default=None)
    args = parser.parse_args()
    os.makedirs('./files',exist_ok=True)

    os.makedirs('./imgs',exist_ok=True)

    dataset_camera_parameter_path = os.path.join(args.eg3d_dataset_dir,'dataset.json')
    with open(dataset_camera_parameter_path, 'rb') as f:
        json_data = json.load(f, object_pairs_hook=OrderedDict)
    labels = json_data["labels"]
    
    x_data = []
    y_data = []
    pbar = tqdm(total=len(labels))
    for i in range(len(labels)):
    
        pbar.update(1)
        camera = labels[i][1]
    
        camera = np.array(camera)[:16]
    
        camera = np.reshape(camera,(4,4))
        co = camera[:, 3:4]
        x = co[0,0]
        y = co[1, 0]
        z = co[2, 0]
    
        theta,phi = convert_cart_to_spherical(np.reshape(co[:3, :],(1,3)))




        x_data.append(theta)
        y_data.append(phi)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    np.save('./files/camera_thetas.npy',x_data)
    np.save('./files/camera_phis.npy',y_data)
    
    print(np.array(x_data).shape, np.array(y_data).shape)
    
    x_original = x_data[:139914]
    y_original = y_data[:139914]

    x_lpff = x_data[139914:]
    y_lpff = y_data[139914:]

    plt.figure(figsize=(8, 8))
    plt.scatter(
        np.array(x_original) / np.pi * 180, np.array(y_original) / np.pi * 180,  # 纵坐标
        c='black', s=2, alpha=0.5, label='FFHQ', rasterized=True)

    plt.scatter(
        np.array(x_lpff) / np.pi * 180, np.array(y_lpff) / np.pi * 180,  # 纵坐标
        c='#40A8C4', s=2, alpha=0.5, label='LPFF', rasterized=True)

    ax = plt.gca()
    ax.set_aspect(1)

    plt.xlabel(r"$\mathit{\theta}$", fontsize=35)
    plt.ylabel(r"$\mathit{\phi}$", rotation=0, fontsize=35)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_xticklabels(
        [r'$0^\circ$', r'$30^\circ$', r'$60^\circ$', r'$90^\circ$', r'$120^\circ$', r'$150^\circ$', r'$180^\circ$'],
        rotation=0, fontsize=25)
    ax.set_yticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_yticklabels(
        [r'$0^\circ$', r'$30^\circ$', r'$60^\circ$', r'$90^\circ$', r'$120^\circ$', r'$150^\circ$', r'$180^\circ$'],
        rotation=0, fontsize=25)
    plt.xlim(0, 180)
    plt.ylim(0, 180)
    plt.legend(loc='upper right', fontsize=16, markerscale=10)  # 显示图例
    plt.tight_layout(pad=0.4, w_pad=0.4, h_pad=0.4)
    # 显示所绘图形
    plt.savefig('./imgs/ffhq-lpff.png', dpi=100, format='png',
                bbox_inches='tight')
    plt.show()


