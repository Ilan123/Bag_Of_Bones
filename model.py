from config import *
import numpy as np


def random_shuffle(data, split=0.7, random_seed=42):
    np.random.seed(random_seed)
    split_ind = int(data.shape[0]*split)
    mask = np.zeros(data.shape[0])
    mask[0:split_ind] = 1
    np.random.shuffle(mask)
    return data[mask==1], data[mask==0]

def save_model(model, dest):
    print("Saving model.")
    from joblib import dump
    dump(clf, dest)

if __name__ == '__main__':
    c = conf()

    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    print("Loading dataframe...")
    dataframe_forearm = np.load(c.images_to_train_paths_file_dest.format('hists'))
    labels_forearm = np.load(c.images_to_train_paths_file_dest.format('labels'))
    print("Found {} images hists".format(len(dataframe_forearm)))

    if c.split_data:
        print("Splitting the data to train and test.")
        dataframe_forearm_train, dataframe_forearm_test = random_shuffle(dataframe_forearm, split=c.split_ratio,
                                                                         random_seed=c.random_seed)
        labels_forearm_train, labels_forearm_test = random_shuffle(labels_forearm, split=c.split_ratio,
                                                                   random_seed=c.random_seed)
        print("Splitted to train set with {} images hists".format(len(dataframe_forearm_train)))
        print("Splitted to test set with {} images hists".format(len(dataframe_forearm_test)))
    else:
        print("Loading test dataframe....")

        dataframe_forearm_train = dataframe_forearm
        labels_forearm_train = labels_forearm
        dataframe_forearm_test = np.load(c.images_to_val_paths_file_dest.format('hists'))
        labels_forearm_test = np.load(c.images_to_val_paths_file_dest.format('labels'))

    print("Starting to train the model...")
    clf = svm.SVC(kernel=c.SVM_kernel)
    clf.fit(dataframe_forearm_train, labels_forearm_train)
    print("Finished to train the model.")
    pred_forearm_test = clf.predict(dataframe_forearm_test)
    score = accuracy_score(y_true=labels_forearm_test, y_pred=pred_forearm_test)
    conf_mat = confusion_matrix(y_true=labels_forearm_test, y_pred=pred_forearm_test, normalize='all')
    print("The model accuracy on the test data is : {}".format(score))
    print("The confusion matrix:\n{}".format(conf_mat))
    print("Saving the model to: {}".format(c.model_weights_file_dest))
    print("The model Sensitivity: {}".format(conf_mat[0, 0]/np.sum(conf_mat, axis=1)[0]))
    print("The model Specificity: {}".format(conf_mat[1, 1]/np.sum(conf_mat, axis=1)[1]))
    print("The model Precision: {}".format(conf_mat[0, 0]/np.sum(conf_mat, axis=0)[0]))
    print("The model Negative Predictive Value: {}".format(conf_mat[1, 1]/np.sum(conf_mat, axis=0)[1]))
    save_model(model=clf, dest=c.model_weights_file_dest)

