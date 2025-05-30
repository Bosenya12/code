import pickle
import numpy as np
from numpy import linalg as la
from scipy.ndimage import zoom
# from radioaug import *

idx_size = 50

def load_data(filename=r'datasets/RML2016.10a_dict.pkl', train_idx_size = idx_size):
    Xd = pickle.load(open(filename,'rb'),encoding='latin') # Xd(1cd 20W,2,128) 11calss*20SNR*6000samples  (mode, snr):1000*2*128
    mods,snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1] ] # mods['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'],  snrs[-20:2:20]
    X = []
    lbl = []  # label (mode,snr)
    train_idx = []
    val_idx = []
    test_idx = []
    np.random.seed(2016)
    a=0  

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])     # ndarray(6000,2,128)
            for i in range(Xd[(mod,snr)].shape[0]):  # 1000
                lbl.append((mod,snr))
            train_idx += list(np.random.choice(range(a * 1000, (a + 1) * 1000), size=train_idx_size, replace=False))
            val_idx += list(np.random.choice(list(set(range(a * 1000, (a + 1) * 1000)) - set(train_idx)), size=800-train_idx_size, replace=False))
            a+=1

    X = np.vstack(X)   # (220000, 2, 128)

    test_idx += list(set(range(0, len(X))) - set(train_idx) - set(val_idx))
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    Y_train = np.array(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))    
    Y_val = np.array(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = np.array(list(map(lambda x: mods.index(lbl[x][0]),test_idx)))

    # ######################################### Translocation
    # trans_X, trans_Y = Translocation_signal(X_train, Y_train, idx_size)
    #
    # ######################################### Ring
    # ring_X, ring_Y = Ring_signal(X_train, Y_train)
    #
    ######################################### Breakage
    # brkg_X, brkg_Y = Breakage_siganl(X_train, Y_train)
    # # Uncomment the following two lines of code only when Breakage is applied alone.
    # X_train = np.concatenate([X_train, np.vstack(brkg_X)], 0)
    # Y_train = np.concatenate([Y_train, brkg_Y], 0)

    # ######################################### Inversion
    # inv_X, inv_Y = Inversion_signal(X_train, Y_train)
    #
    # ######################################### Terminal Deletion
    # termdel_X, termdel_Y = Terminal_Deletion_signal(X_train, Y_train)
    #
    # ######################################### Interstitial Deletion
    # intdel_X, intdel_Y = Interstitial_Deletion_signal(X_train, Y_train)
    #
    # ######################################### Gaussian_Noise
    # noise_X, noise_Y = Gaussian_Noise_signal(X_train, Y_train)
    #
    # # ######################################### Filp
    # # flip_X, flip_Y = Flip_signal(X_train, Y_train)
    # #
    # # ######################################### Rotation
    # # rotation_X, rotation_Y = Rotation_signal(X_train, Y_train)
    #
    # ######################################### Filp and Rotation
    # flip_rotation_X, flip_rotation_Y = Filp_and_Rotation_signal(X_train, Y_train)
    #
    #
    # # Uncomment the following two lines only when all six augmentation methods are used at the same time.
    # X_train = np.concatenate([X_train, np.vstack(trans_X), np.vstack(ring_X), np.vstack(brkg_X), np.vstack(inv_X),
    #                           np.vstack(termdel_X), np.vstack(intdel_X), np.vstack(noise_X), np.vstack(flip_rotation_X)], 0)
    # Y_train = np.concatenate([Y_train, trans_Y, ring_Y, brkg_Y, inv_Y, termdel_Y, intdel_Y, noise_Y, flip_rotation_Y], 0)


    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    return (mods, snrs, lbl), (X_train,Y_train), (X_test,Y_test), (train_idx, test_idx)

if __name__ == '__main__':
    (mods, snrs, lbl), (X_train,Y_train), (X_test,Y_test), (train_idx,test_idx) = load_data()

