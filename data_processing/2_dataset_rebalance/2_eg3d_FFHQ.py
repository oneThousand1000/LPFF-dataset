

import json
import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eg3d_dataset_dir", type=str, default=None)
    args = parser.parse_args()
    fnames = []

    thetas_all = np.load('./files/camera_thetas.npy')
    phis_all = np.load('./files/camera_phis.npy')

    thetas_ffhq = thetas_all[:139914]
    phis_ffhq = phis_all[:139914]

    eg3d_dataset_dir = args.eg3d_dataset_dir
    pbar = tqdm.tqdm(total=139914)
    for i in range(139914):
        pbar.update(1)

        fnames.append(f'{int(i // 1000):05d}/img{i:08d}.png')




    print('\n',len(fnames),'samples!')
    data = json.dumps(fnames, indent=1)
    with open(os.path.join(eg3d_dataset_dir,f'FFHQ.json'), "w", newline='\n') as f:
        f.write(data)

    plt.figure(figsize=(8, 8))
    plt.scatter(
        thetas_ffhq/ np.pi * 180, phis_ffhq / np.pi * 180,  # 纵坐标
        c='black', s=2, alpha=0.5, label='FFHQ', rasterized=True)


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
    plt.savefig('./imgs/ffhq.png', dpi=100, format='png',
                bbox_inches='tight')
    plt.savefig('./imgs/ffhq.pdf', dpi=100, format='pdf',
                bbox_inches='tight')
    plt.show()
