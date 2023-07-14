import numpy as np
import os
import torch
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    root = args.data_dir
    raw_image_dir = os.path.join(root, 'raw')
    re_aligned_dir = os.path.join(root, 'realign')
    
    in_root = os.path.join(re_aligned_dir,'epoch_20_000000')
    seleted_dir = os.path.join(re_aligned_dir,'debug')

    out_root = os.path.join(re_aligned_dir,'camera_parameters')
   
    os.makedirs(out_root,exist_ok=True)

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


    debug_imgs = sorted([x for x in os.listdir(seleted_dir) if x.endswith(".png")])

    mode = 1  # 1 = IDR, 2 = LSX
    outAll = {}

    for debug_img in debug_imgs:
        src_filename = debug_img.replace('png','npy')

        src = os.path.join(in_root, src_filename)
        camera_parameters_dst = os.path.join(out_root, src_filename)
        if os.path.exists(camera_parameters_dst):
            continue
        dict_load = np.load(src, allow_pickle=True)

        angle = dict_load.item()['angle']
        trans = dict_load.item()['trans'][0]
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

        focal = 2985.29  # = 1015*1024/224*(300/466.285)#
        pp = 512  # 112
        w = 1024  # 224
        h = 1024  # 224

        count = 0
        K = np.eye(3)
        K[0][0] = focal
        K[1][1] = focal
        K[0][2] = w / 2.0
        K[1][2] = h / 2.0
        K = K.tolist()

        Rot = np.eye(3)
        Rot[0, 0] = 1
        Rot[1, 1] = -1
        Rot[2, 2] = -1
        pose[:3, :3] = np.dot(pose[:3, :3], Rot)
        #print('pose: ', pose)
        pose = pose.tolist()
        out = {}
        out["intrinsics"] = K
        out["pose"] = pose
        out["angle"] = (angle * [1, -1, 1]).flatten().tolist()
        
        

        camera_parameters = np.concatenate([np.reshape(pose, (16)),
                                            np.array([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0])

                                            ])
        outAll[src_filename.split('.')[0]] = camera_parameters.tolist()
        #print('save camera_parameters to', camera_parameters_dst)
        np.save(camera_parameters_dst, camera_parameters)

    if mode == 1:
        dst = os.path.join(out_root, "camera_parameters.json")
        with open(dst, "w") as outfile:
            json.dump(outAll, outfile, indent=4)
