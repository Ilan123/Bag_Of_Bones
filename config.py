class conf:
    def __init__(self):
        self.is_creating_descriptors_list = False
        self.is_creating_kmeans_obj = False
        self.is_using_PCA = True
        self.is_creating_new_PCAed_des = True
        self.create_hists = True

        ### Paths ###
        self.images_paths_file_dest = "../forearm_paths.csv"
        self.des_file_path = "../forearm_des.npy"
        self.des_PCAed_file_path = "../forearm_des_PCAed40.npy"
        self.kmeans_centroids_PCA_dim_file_dest = "../centroids64_forearm_PCAed_batch100_k20000_.npy"
        self.kmeans_centroids_file_dest = "../centroids64_forearm_batch100_k20000_.npy"
        self.images_hist_file_dest = "../forearm_k20000_{}.npy"
        self.model_weights_file_dest = "../SVM_model_k20000_.joblib"
        self.images_to_train_paths_file_dest = "../forearm_k20000_train_{}.npy"
        self.images_to_val_paths_file_dest = "../forearm_k20000_val_{}.npy"


        ### l0 func & get_features settings ###
        self.gf_start_from_previous_file = False
        self.l0_with_smooth = 1
        self.l0_wind_s = 20
        self.l0_lmd = 0.15

        ### Kmeans Settings ###
        self.kmeans_num_of_clusters = 20000
        self.kmeans_batch_size = 100

        ### PCA Settings ###
        self.PCA_num_of_dimnetions = 40

        ### calculate_hists_dataframe ###
        self.chd_start_from_previous_file = False

        ### model ####
        self.SVM_train_new_model = True
        self.SVM_kernel = 'rbf'
        self.split_data = True
        self.split_ratio = 0.7
        self.random_seed = 42
