from config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import cv2
from scipy.signal import convolve2d
from scipy.fftpack import fft, ifft, fft2, ifft2, fftshift, ifftshift
from sklearn.decomposition import PCA
from scipy import spatial
from sklearn.neighbors import KDTree


#%matplotlib inline


############# START: l0 func and sub funcs #############
def circulantshift2_x(xs, h):
    return np.hstack([xs[:, h:], xs[:, :h]] if h > 0 else [xs[:, h:], xs[:, :h]])

def circulantshift2_y(xs, h):
    return np.vstack([xs[h:, :], xs[:h, :]] if h > 0 else [xs[h:, :], xs[:h, :]])

def circulant2_dx(xs, h):
    return (circulantshift2_x(xs, h) - xs)

def circulant2_dy(xs, h):
    return (circulantshift2_y(xs, h) - xs)

def get_im(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = np.array(img).astype('uint8')
    return img

"""

"""
def l0_gradient_minimization_2d(I, lmd, beta_max, beta_rate=2.0, max_iter=30, return_history=False):
    S = np.array(I)

    # prepare FFT
    F_I = fft2(S, axes=(0, 1))
    Ny, Nx = S.shape[:2]
    D = S.shape[2] if S.ndim == 3 else 1
    dx, dy = np.zeros((Ny, Nx)), np.zeros((Ny, Nx))
    dx[int(Ny / 2), int(Nx / 2 - 1):int(Nx / 2 + 1)] = [-1, 1]
    dy[int(Ny / 2 - 1):int(Ny / 2 + 1), int(Nx / 2)] = [-1, 1]
    F_denom = np.abs(fft2(dx)) ** 2.0 + np.abs(fft2(dy)) ** 2.0
    if D > 1: F_denom = np.dstack([F_denom] * D)

    S_history = [S]
    beta = lmd * 2.0
    hp, vp = np.zeros_like(S), np.zeros_like(S)
    for i in range(max_iter):
        # print(i)
        # with S, solve for hp and vp in Eq. (12)
        hp, vp = circulant2_dx(S, 1), circulant2_dy(S, 1)
        if D == 1:
            mask = hp ** 2.0 + vp ** 2.0 < lmd / beta
        else:
            mask = np.sum(hp ** 2.0 + vp ** 2.0, axis=2) < lmd / beta
        hp[mask] = 0.0
        vp[mask] = 0.0

        # with hp and vp, solve for S in Eq. (8)
        hv = circulant2_dx(hp, -1) + circulant2_dy(vp, -1)
        S = np.real(ifft2((F_I + (beta * fft2(hv, axes=(0, 1)))) / (1.0 + beta * F_denom), axes=(0, 1)))

        # iteration step
        if return_history:
            S_history.append(np.array(S))
        beta *= beta_rate
        if beta > beta_max: break

    if return_history:
        return S_history

    return S

############# END: l0 func and sub funcs #############

############# Start: Images preprocessing and descriptors find funcs #############
"""
Applying CLAHE - Contrast Limited Adaptive Histogram Equalization
and convert the unput to uint8
wind_s - CLAHE window size
"""
def stab_hist(im, wind_s):
    CLAHE_obj = cv2.createCLAHE(clipLimit=10, tileGridSize=(wind_s, wind_s))
    img_Clahe = CLAHE_obj.apply(im).astype('uint8')
    return img_Clahe

"""
Activing stab_hist function on the image,
along with L0.
Converting the output to uint8
wind_s - CLAHE window size
lmd - L0 Lamda parameter (the lower lmd, the more none zero gradients (edges preserved)).
"""
def prepro(im, with_smooth, wind_s, lmd):
    stab_im = stab_hist(im, wind_s)
    if with_smooth == 1:
        stab_im = l0_gradient_minimization_2d(stab_im, lmd, 100000)
    return stab_im.astype('uint8')

"""
Extracting descriptors from the preprocessed images in the src file,
and saving all the images descriptors together in the dest npy file.

Reading the images paths one by one from the csv paths list, 
applying the preprocessing stages: CLAHE and L0,
extracting the descriptors and saving them together.

Keyword arguments:
    src -- an csv file with the images paths
    dest -- file to save all the descriptors that was extracted
    with_smooth -- set to 1 to apply L0
    lmd -- L0 lamda
    last_iter_dest -- A file which document the function last iteration that was saved, used to start the alghorthem,
                        from a middle of a precious running.
    start_from_previous_file -- boolean parameter, determnt weather to create a new dest file, or proceed from the 
                        previous one
"""
def get_features(src, dest='../test.npy', with_smooth=1, wind_s=20, lmd=0.15, last_iter_dest='../gf_last_it.npy', start_from_previous_file = False):
    fast = cv2.FastFeatureDetector_create()
    # run on file path and read images for each im_path
    print("Trying to read the {} file...".format(src))
    data_paths_dataframe = pd.read_csv(src, header=None)
    print("Found {} Paths".format(len(data_paths_dataframe)))
    starting_index = 0

    if start_from_previous_file:
        print("A previous descriptors list was mention, reading the file")
        des_lst = np.load(dest)
        starting_index = np.load(last_iter_dest)
        print("Successfully read the descriptors file: {}, and the last iter file: {}".format(dest, last_iter_dest))
    else:
        print("No previous extraction was mention, creating new descriptors list")
        des_lst = np.empty((0, 64))

    print("Starting to preprocessing and to extracting descriptors from the images...".format(src))
    for i in range(starting_index, data_paths_dataframe.shape[0]):
        relative_im_path = "../{}".format(data_paths_dataframe.iloc[i, 0])
        # print(relative_im_path)
        im = get_im(relative_im_path)
        # displayImage(im, "im")
        im = prepro(im, with_smooth, wind_s, lmd)
        # displayImage(im, "im_prepro")
        #kp = fast.detect(im, None)
        br = cv2.BRISK_create()
        #kp, des = br.compute(im, kp)
        kp, des = br.detectAndCompute(im, None)
        des_lst = np.vstack((des_lst, des))
        if i % 50 == 0 or (i == data_paths_dataframe.shape[0] - 1):

            np.save(last_iter_dest, i)
            np.save(dest, des_lst)

            print("Extract and saved features from {} images so far".format(i))

    print("Finished to extract and save features from the images")
    return

"""
Extracting descriptors from a single image.
Applying the preprocessing stages: CLAHE and L0,
and extracting the descriptors using BRISK from the result.
return key points and descriptors list
"""
def get_im_descriptors(im, with_smooth=1, wind_s=20, lmd=0.15, brisk_obj=None):
    im = prepro(im, with_smooth, wind_s, lmd)
    #fast = cv2.FastFeatureDetector_create()
    # displayImage(im, "im_prepro")

    if not brisk_obj:
        brisk_obj = cv2.BRISK_create()
    kp, des = brisk_obj.detectAndCompute(im, None)  # note: no mask here!

    return kp, des

############# END: Images preprocessing and descriptors find funcs #############

############# START: Kmeans, PCA and hist creation #############
def create_kmeans_obj(data, k=10000, batch_size=10000):
    # k = np.size(species) * 10

    # batch_size = np.size(os.listdir(img_path)) * 3
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(data)
    return kmeans

"""
Creating a bag of visual words histogram according to the centers tree, that describes the images.
Keyword arguments:
    img_path
    centers_tree - an Scipy.spatial.cKDTree Object, which is used to classify the image descriptors.
    num_of_centers - that the tree was created with
    brisk_obj - it is possible to pre create the brisk object for efficiency
    
Returns:
    np.array(dtype=int) idx - a vector of labeled descriptors
     np.array(dtype=int) hist - histogram with size num_of_centers the describe the image.
"""
def predict_image(img_path, centers_tree, num_of_centers, brisk_obj=None):
    im = get_im(img_path)
    kp, des = get_im_descriptors(im, brisk_obj)
    # print("des.shape: ", des.shape, "\n")
    idx = np.array(centers_tree.query(des)[1])
    # dist, idx = centers_tree.query(des, k=1) # this can be used if we are using sklearn k nearest neighbors instead.
    # creating the histogram:
    hist = np.zeros(num_of_centers)
    labels_indexes, labels_counts = np.unique(idx, return_counts=True)
    hist[labels_indexes.astype(int)] = labels_counts

    return idx, hist


"""
Creating a bag of visual words histogram according to the given KMeans model, that describes the images.
Keyword arguments:
    img_path
    kmeans - KMeans sklearn object
    brisk_obj - it is possible to pre create the brisk object for efficiency

Returns:
    np.array(dtype=int) idx - a vector of labeled descriptors
    np.array(dtype=int) hist - histogram with size num_of_centers the describe the image.
"""
def predict_kmeans(img_path, kmeans, brisk_obj=None):
    im = get_im(img_path)
    kp, des = get_im_descriptors(im, brisk_obj)
    print("des.shape: ", des.shape, "\n")
    idx = kmeans.predict(des)
    # histo[idx] += 1/nkp

    # creating the histogram:
    print("Max value: ", np.max(idx))
    hist = np.zeros(1000)
    labels_indexes, labels_counts = np.unique(idx, return_counts=True)
    hist[labels_indexes.astype(int)] = labels_counts

    return idx, hist


def show_hist(hist):
    f, ax = plt.subplots(sharex='col', sharey='row', figsize=(24, 8))
    plt.hist(hist, bins=1000, range=[0, 1000])
    plt.show()

"""
Applying PCA on the given list, to reduce it axis=1 dimensions to X.
"""
def prepareX_D(X,data):
    pca = PCA(n_components=X)
    pca.fit(data)
    data_out = pca.transform(data)

    return data_out

"""
Given a data and labels of the data,
this creates centroids according the the labels.
The centroids are the mean vector of the vectors that belongs to the same cluster.

:returns np.array(centroids_original_dimension)
"""
def create_original_dimention_centroids(data, labels):
    print("Got {} unique labels".format(np.unique(labels).shape))
    centroids_original_dimention = []
    labels_type, label_count = np.unique(labels, return_counts=True)
    for i in range(len(labels_type)):
        centroids_original_dimention.append(data[labels == labels_type[i]].sum(axis=0) / label_count[i])
        if (i % 100 == 0 or i == len(labels_type) - 1):
            print("Finished the {} iteration".format(i))

    # un, counts = np.unique(labels, return_counts=True)

    return np.array(centroids_original_dimention)

"""
Creating a list of histogram that describe the images in the given path,
according to the given Scipy.spatial.cKDTree Object.
Keyword arguments:
    images_path -- an csv file with the images paths
    dest -- file to save all the histograms of the images
    centers_tree -- Scipy.spatial.cKDTree Object that used to classify the images descriptors
    last_iter_dest -- A file which document the function last iteration that was saved, used to start the alghorthem,
                        from a middle of a precious running.
    start_from_previous_file -- boolean parameter, determnt weather to create a new dest file, or proceed from the 
                        previous one
    
returns:
    np.array(dimensions=2,dtype=int) -- a list of the images histograms vectors, coresponding to the order of the images
                            in the images_path file.
    np.array(dtype=int) -- a vectors containing the images labels (0, 1)
"""
def calculate_hists_dataframe(images_path, dest, centers_tree, num_of_centers, last_iter_dest='../chd_last_it.npy', start_from_previous_file = False):
    dataframe = []
    labels = []
    print("Calculate_hists_dataframe")
    print("loading images paths from: {}".format(images_path))
    data_paths_dataframe = pd.read_csv(images_path, header=None)
    print("Found {} images".format(len(data_paths_dataframe)))

    dest_labels = dest.format('hists')
    dest_data = dest.format('labels')
    br = cv2.BRISK_create()
    starting_index = 0
    if start_from_previous_file:
        starting_index = np.load(last_iter_dest)

    for i in range(starting_index, data_paths_dataframe.shape[0]):
        relative_im_path = "../{}".format(data_paths_dataframe.iloc[i, 0])

        des_labels_indexes, hist = predict_image(img_path=relative_im_path, centers_tree=centers_tree,
                                                 num_of_centers=num_of_centers, brisk_obj=br)
        dataframe.append(hist)
        labels.append(0) if relative_im_path.find('negative') != -1 else labels.append(1)
        if i % 50 == 0 or i == data_paths_dataframe.shape[0] - 1:
            np.save(dest_labels, dataframe)
            np.save(dest_data, labels)
            np.save(last_iter_dest, i)
            print("Finished the {} iteration".format(i))
            print("Saved to: {}, and {}".format(dest_data, dest_labels))

    return dataframe, labels

############# END: Kmeans, PCA and hist creation #############

if __name__ == '__main__':
    c = conf()

    if c.is_creating_descriptors_list:
        get_features(src=c.images_paths_file_dest, dest=c.des_file_path, with_smooth=c.l0_with_smooth,
                     wind_s=c.l0_wind_s, lmd=c.l0_lmd, start_from_previous_file=c.gf_start_from_previous_file)

        print("Trying to read the des_file...")
        try:
            test_read = np.load(c.des_file_path)
            print('A file with {} dimensions was found'.format(test_read.shape))
            test_read = None
        except IOError:
            print("ERROR: (IOError) Failed to read the des_file!")
            exit(-1)

    if c.is_creating_kmeans_obj:
        print("Loading descriptors list for kmeans...")
        data = np.load(c.des_file_path)
        print("Successfully read the descriptors list, and found: {} (shape) descriptors".format(data.shape))

        if c.is_using_PCA:
            if c.is_creating_new_PCAed_des:
                print("Using PCA to reduce dimensions to: {}".format(c.PCA_num_of_dimnetions))
                data_PCAed = prepareX_D(X=c.PCA_num_of_dimnetions, data=data)
                np.save(c.des_PCAed_file_path, data_PCAed)
                print("Finished to reduce dimensions, and saved to: {}".format(c.des_PCAed_file_path))

            else:
                print("Loading PCAed descriptors from {}".format(c.des_PCAed_file_path))
                data_PCAed = np.load(c.des_PCAed_file_path)
                print("Successfully read the PCAed descriptor list, and found: {} (shape) descriptors".format(
                    data_PCAed.shape))

            print("Start training a kmeans model with clusters: {}  batch_size: {}".format(c.kmeans_num_of_clusters,
                                                                                           c.kmeans_batch_size))
            kmeans_obj = create_kmeans_obj(data=data_PCAed, k=c.kmeans_num_of_clusters, batch_size=c.kmeans_batch_size)
            print("Finished to kmeans training.")
            centroids_PCAed = kmeans_obj.labels_

            print("Saving the kmeans PCAed centroids to: {}".format(c.kmeans_centroids_PCA_dim_file_dest))
            np.save(c.kmeans_centroids_PCA_dim_file_dest, centroids_PCAed)
            centroids = create_original_dimention_centroids(data=data, labels=centroids_PCAed)

        else:
            kmeans_obj = create_kmeans_obj(data=data, k=c.kmeans_num_of_clusters, batch_size=c.kmeans_batch_size)
            centroids = kmeans_obj.labels_

        print("Saving the kmeans centroids to: {}".format(c.kmeans_centroids_file_dest))
        np.save(c.kmeans_centroids_file_dest, centroids)

    else:
        print("Loading the centroids from: {}".format(c.kmeans_centroids_file_dest))
        centroids = np.load(c.kmeans_centroids_file_dest)

    if c.create_hists:
        centers_tree = spatial.cKDTree(centroids)
        #centers_tree = KDTree(centroids)
        dataframe_forearm, labels_forearm = calculate_hists_dataframe(images_path=c.images_paths_file_dest,
                                                                      dest=c.images_hist_file_dest,
                                                                      centers_tree=centers_tree,
                                                                      num_of_centers=c.kmeans_num_of_clusters,
                                                                      start_from_previous_file=
                                                                      c.chd_start_from_previous_file)


