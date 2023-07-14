import numpy as np
import matplotlib
from tqdm import tqdm
import scipy.stats as st
import pickle



if __name__ == '__main__':
    matplotlib.use('TkAgg')
    rs = np.random.RandomState(15)

    thetas = np.load('./files/camera_thetas.npy')[:139914]
    phis = np.load('./files/camera_phis.npy')[:139914]
    print(thetas.shape, phis.shape)

    print(np.min(thetas),np.max(thetas))
    print(np.min(phis),np.max(phis))

    values = np.vstack([thetas, phis])
    kernel = st.gaussian_kde(values)

    with open('./files/kde-ffhq-fixed.pkl', 'wb') as f:  # open file with write-mode
        picklestring = pickle.dump(kernel, f)




