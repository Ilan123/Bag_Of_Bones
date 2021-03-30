from config import *
from preprocessor import *
import numpy as np
import pandas as pd


def load_model(dest):
    from joblib import load
    print("Loading previous model...")
    return load(dest)

def predict_a_single_image(image_path, model):
    bag_words = np.load(c.kmeans_centroids_file_dest)
    centers_tree = spatial.KDTree(centroids)
    des_labels_indexes, hist = predict_image(image_path, centers_tree=centers_tree,
                                             num_of_centers=c.kmeans_num_of_clusters)

    return model.predict(np.array([hist]))[0]

if __name__ == '__main__':
    c = conf()
    image_to_predict_path = ""
    is_prediction_single_image = False
    is_test = True

    clf = load_model(c.model_weights_file_dest)

    if is_prediction_single_image:
        im_pred = predict_a_single_image(image_path=image_to_predict_path, model=clf)
        print("The prediction is: {}".format("Negative" if im_pred == 0 else "Positive"))

    else:
        dataframe_forearm = np.load(c.images_to_val_paths_file_dest.format('hists'))

        pred_labels = clf.predict(dataframe_forearm)

        if is_test:
            from sklearn.metrics import accuracy_score
            from sklearn.metrics import confusion_matrix
            labels_forearm = np.load(c.images_to_val_paths_file_dest.format('labels'))
            score = accuracy_score(y_true=labels_forearm, y_pred=pred_labels)
            conf_mat = confusion_matrix(y_true=labels_forearm, y_pred=pred_labels)
            print("The model accuracy on the test data is : {}".format(score))
            print("The confusion matrix:\n{}".format(conf_mat))

        else:
            images_paths = pd.read_csv(c.images_paths_file_dest, header=None)
            images_paths["label"] = pred_labels
            images_paths.to_csv('../pred.csv', index=False)
