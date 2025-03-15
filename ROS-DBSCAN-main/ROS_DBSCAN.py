
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from kneed import KneeLocator
import Measurements as M
from Style import Configure as Conf


def find_eps(x_train, min_pnts):
    nbrs = NearestNeighbors(n_neighbors=min_pnts - 1).fit(x_train)
    distances, indices = nbrs.kneighbors(x_train)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    # plt.plot(distances)
    # plt.show()
    from scipy.signal import savgol_filter
    sg_distances = savgol_filter(distances, 51, 2)
    # plt.plot(range(1, len(sg_distances) + 1), sg_distances, range(1, len(sg_distances) + 1), distances)
    # plt.show()
    kneedle = KneeLocator(range(1, len(sg_distances) + 1),  # x values
                          sg_distances,  # y values
                          S=0,  # measure of how many “flat” points we expect to see in the unmodified data curve
                          curve="convex",  # parameter from figure concave/convex
                          online=True,
                          direction="increasing")  # parameter from figure
    # kneedle = KneeLocator(range(int(len(sg_distances)*0.7), len(sg_distances) + 1),  # x values
    #                       sg_distances[int(len(sg_distances)*0.7) - 1:],  # y values
    #                       S=0,  # measure of how many “flat” points we expect to see in the unmodified data curve
    #                       curve="convex",  # parameter from figure concave/convex
    #                       online=True,
    #                       direction="increasing")  # parameter from figure
    # kneedle.plot_knee_normalized()
    ## print(kneedle.elbow)
    ## print(kneedle.knee_y)
    ## kneedle.plot_knee()
    return kneedle.knee_y


def concat_datasets(datasets):
    trainings, positives, negatives = datasets
    training = pd.concat(trainings, ignore_index=True)
    positive = pd.concat(positives, ignore_index=True)
    negative = pd.concat(negatives, ignore_index=True)
    x_train = training
    x_test = pd.concat([positive, negative], ignore_index=True)
    x_dfs = pd.concat([training, positive, negative], ignore_index=True)
    return x_train, x_test, x_dfs


def run_dbscan_n_predict(datasets, my_eps=0, min_group=0):
    def predict(class_trains, sample, max_dist):
        min_dist = np.min(cdist(class_trains, [sample]))
        if max_dist >= min_dist:
            return 1
        return 0

    Xtrain, Xtest, Xdfs = concat_datasets(datasets)
    if min_group == 0:
        min_group = Xdfs.shape[1] + 1
    if my_eps == 0:
        my_eps = find_eps(x_train=Xtrain, min_pnts=min_group)  # datasets[0] == trainings
        my_eps += 0.0001
    Xtrain = Xtrain.to_numpy()
    Xtest = Xtest.to_numpy()
    eps_calibration = True
    predictions = []
    i_pred = []
    predicts = []
    while(eps_calibration):
        labels_ = DBSCAN(eps=my_eps, min_samples=min_group).fit_predict(Xtrain)
        predicts = labels_
        fp_labels = [i for i in labels_ if i < 0]
        if len(fp_labels)/len(labels_) > 0.1:
            my_eps += 0.04
        else:
            eps_calibration = False


    for i in range(0, len(predicts), 1):
        if predicts[i] >= 0:
            predictions.append(1)
            i_pred.append(i)
        else:
            predictions.append(0)
    ##print("----------------------- EPS =     " + str(my_eps) + "    min group =    " + str(min_group) +
    ##      "  -----------------------")
    # update the datasets_preds
    pos_train = [Xtrain[i] for i in i_pred]
    if len(i_pred) < 1:
        print("Error")
    for sample in Xtest:
        predictions.append(predict(pos_train, sample, my_eps))

    dfs, n_dfs = convert_concat_to_datasets_n_prediction(Xdfs, predictions, datasets)
    return dfs, n_dfs


def convert_concat_to_datasets_n_prediction(Xdf, Ydf, datasets):
    count = 0
    trainings, positives, negatives = datasets
    new_trainings_dataset = []
    new_trainings_predicts = []
    for train in trainings:
        dataset = Xdf.iloc[count:train.shape[0]+count, :]
        predicts = Ydf[count:train.shape[0]+count]
        new_trainings_dataset.append(dataset)
        new_trainings_predicts.append(predicts)
        count += train.shape[0]
    new_positives_dataset = []
    new_positives_predicts =[]
    for pos in positives:
        dataset = Xdf.iloc[count:pos.shape[0]+count, :]
        predicts = Ydf[count:pos.shape[0]+count]
        new_positives_dataset.append(dataset)
        new_positives_predicts.append(predicts)
        count += pos.shape[0]
    new_negatives_dataset = []
    new_negatives_predicts = []
    for neg in negatives:
        dataset = Xdf.iloc[count:neg.shape[0]+count, :]
        predicts = Ydf[count:neg.shape[0]+count]
        new_negatives_dataset.append(dataset)
        new_negatives_predicts.append(predicts)
        count += neg.shape[0]
    p_datasets = new_trainings_predicts, new_positives_predicts, new_negatives_predicts
    n_datasets = new_trainings_dataset, new_positives_dataset, new_negatives_dataset
    return n_datasets, p_datasets
