import numpy as np
import matplotlib
from tqdm import tqdm
import scipy.stats as st
import pickle



if __name__ == '__main__':
    matplotlib.use('TkAgg')
    rs = np.random.RandomState(15)

    thetas = np.load('./files/camera_thetas.npy')
    phis = np.load('./files/camera_phis.npy')
    print(thetas.shape, phis.shape)

    print(np.min(thetas),np.max(thetas))
    print(np.min(phis),np.max(phis))

    values = np.vstack([thetas, phis])
    kernel = st.gaussian_kde(values)

    with open('./files/kde-ffhq-flickr.pkl', 'wb') as f:  # open file with write-mode
        picklestring = pickle.dump(kernel, f)



    density_all = []
    pbar = tqdm(total=thetas.shape[0])
    for i in range(thetas.shape[0]):
        theta = thetas[i]
        phi = phis[i]
        values = np.vstack([theta, phi])
        f = kernel(values)[0]
        #print(f)
        pbar.update(1)
        density_all.append(f)

    print(len(density_all))
    np.save('./files/density_FFHQ_LPFF.npy', np.array(density_all))





