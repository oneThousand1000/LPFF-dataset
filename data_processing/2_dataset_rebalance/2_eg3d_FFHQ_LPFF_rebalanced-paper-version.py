

import json
import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import tqdm

color = {1: '#377EDB', 2: '#85A97E', 3: '#D4A9B8', 4: '#9E406B', 5: '#732B7E', 6: '#DB8039', 7: '#A940A4', 8: '#C8A242', 9: '#10095A', 10: '#CA9BE4', 11: '#7AA8DE', 12: '#7AA8DE', 13: '#B58A5D', 14: '#ED71D9', 15: '#05DE75', 16: '#DDECC8', 17: '#0D5308', 18: '#10095A', 19: '#F5981D', 20: '#053687', 21: '#6BB4A1', 22: '#1EEFFA', 23: '#D3B435', 24: '#BB79E1', 25: '#055C32', 26: '#2B75E3', 27: '#53E03B', 28: '#FA4453', 29: '#194DA4', 30: '#C8A075', 31: '#CF4B23', 32: '#F748E8', 33: '#04E639', 34: '#343ECB', 35: '#A940A4', 36: '#C44C57', 37: '#6DBAB6', 38: '#1D55F7', 39: '#616FD8', 40: '#4C7C1F', 41: '#4C0A59', 42: '#42DF97', 43: '#62009A', 44: '#E91254', 45: '#AEE45F', 46: '#B708DC', 47: '#95F129', 48: '#D9578C', 49: '#B4763B', 50: '#FDD38F', 51: '#D3B14A', 52: '#020187', 53: '#8EDDA5', 54: '#1A9329', 55: '#F8D600', 56: '#393181', 57: '#656C7F', 58: '#2D1789', 59: '#6CDAA4', 60: '#D63E9C', 61: '#9CB8A1', 62: '#871B3B', 63: '#56AA6A'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eg3d_dataset_dir", type=str, default=None)
    args = parser.parse_args()
    fnames = []

    thetas_all = np.load('./files/camera_thetas.npy')
    phis_all = np.load('./files/camera_phis.npy')
    all_densitys = np.load('./files/density_FFHQ_LPFF.npy')

    eg3d_dataset_dir = args.eg3d_dataset_dir
    pbar = tqdm.tqdm(total=thetas_all.shape[0])

    vis_theta = []
    vis_phi = []
    for i in range(8):
        vis_theta.append([])
        vis_phi.append([])

    with open('./files/lpff_ffhq_rebal.json','r') as f:
        duplicate_num_dict = json.load(f)
    for i in range(thetas_all.shape[0]):
        pbar.update(1)
        theta = thetas_all[i]
        phi = phis_all[i]
        density = all_densitys[i]
        # duplicate_num = int(min(max(round(4 * 0.06 / density), 1), 4))
        # # if density < 0.01:
        # #     duplicate_num = 7
        # #     count+=1
        # # el
        # if density < 0.02:
        #     duplicate_num = 6
        # elif density < 0.03:
        #     duplicate_num = 5
        duplicate_num = duplicate_num_dict[str(i)]
        vis_theta[duplicate_num].append(theta)
        vis_phi[duplicate_num].append(phi)
        for index in range(int(duplicate_num)):
            fnames.append(f'{int(i // 1000):05d}/img{i:08d}.png')




    print('\n',len(fnames),'samples!')
    data = json.dumps(fnames, indent=1)
    with open(os.path.join(eg3d_dataset_dir,f'FFHQ_LPFF_rebalanced.json'), "w", newline='\n') as f:
        f.write(data)

    plt.figure(figsize=(8, 8))
    for i in range(1, 7):
        plt.scatter(
            np.array(vis_theta[i]) / np.pi * 180, np.array(vis_phi[i]) / np.pi * 180,  # 纵坐标
            c=color[i], s=2, alpha=0.5, label=f'duplicate {i}', rasterized=True)


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
    plt.savefig('./imgs/ffhq_lpff_rebal.png', dpi=100, format='png',
                bbox_inches='tight')
    plt.savefig('./imgs/ffhq_lpff_rebal.pdf', dpi=100, format='pdf',
                bbox_inches='tight')
    plt.show()
