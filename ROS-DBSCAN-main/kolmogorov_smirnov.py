import os
import os.path
import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sb

DATADIR = "data"
EXPERIMENTS = ["real_panda", "sim_panda", "real_turtlebot3", "sim_turtlebot3"]


def normal_datadir(experiment):
    """Returns normal data directory for the experiment.
    """
    return os.path.join(DATADIR, experiment, "normal\\type1")


def abnormal_datadirs(experiment):
    """Returns the list of anomaly names and corresponding data directories
    for the experiment.
    """
    abnormal = os.path.join(DATADIR, experiment, "abnormal")
    datadirs = []
    for entry in os.listdir(abnormal):
        dir = os.path.join(abnormal, entry)
        if os.path.isdir(dir):
            datadirs.append((entry, dir))
    return datadirs


def read_data_file(fname):
    df = pd.read_csv(fname)
    df = df.filter(regex="^Counter\(", axis=1)
    return df


# For KS statistics-based comparison, we only need the set of non-zero counts.
def non_zero_counts(df):
    """Returns a list of non-zero counts in the data frame.
    """
    counts = df.to_numpy()
    return counts[counts > 1].reshape(-1)


def dir_counts(datadir):
    """Returns list of lists of counts for a data directory.
    """
    counts = []
    for entry in os.listdir(datadir):
        path = os.path.join(datadir, entry)
        if os.path.isfile(path):
            counts.append(non_zero_counts(read_data_file(path)))
    return counts


def ks_statistics_of_runs(counts, baseline):
    return np.array([ss.ks_2samp(c, baseline).statistic for c in counts])


def confusion_matrices(normal_counts_train,
                       normal_counts_test, abnormal_counts_test,
                       N=25):
    """Computes confusion matrices for an anomaly. Arguments:
       * normal_counts_train --- normal counts used for choosing the KS threshold.
       * normal_counts_test --- normal counts test set.
       * abnormal_counts_test --- abnormal counts test set.
       * N --- number of points to calculate ROC at.
    Returns:
       confusion matrices for a range of thresholds.
    """
    baseline = np.concatenate(normal_counts_train)

    normal_ks_train = ks_statistics_of_runs(normal_counts_train, baseline)
    # thresholds = np.linspace(normal_ks_train.max(), normal_ks_train.min(), 25)
    tmp = np.sort(normal_ks_train)
    threshold = tmp[int(len(normal_ks_train)*0.9)]
    normal_ks_test = ks_statistics_of_runs(normal_counts_test, baseline)
    abnormal_ks_test = ks_statistics_of_runs(abnormal_counts_test, baseline)

    # just for debug

    print([abnormal_ks_test < threshold])

    cs = []
    c = np.array(
        [[sum(normal_ks_test < threshold), sum(normal_ks_test >= threshold)],
         [sum(abnormal_ks_test < threshold), sum(abnormal_ks_test >= threshold)]])
    cs.append(c)
    x = np.stack(cs, axis=0)
    return x


NFOLDS = 10
print(f"{'experiment/anomaly':34s} | {'mean tpr':8s} | {'std tpr':8s}")
print("-" * 50)
for exp in EXPERIMENTS:
    normal_counts = dir_counts(normal_datadir(exp))
    fold_size = len(normal_counts) // NFOLDS

    for abname, abpath in abnormal_datadirs(exp):
        abnormal_counts = dir_counts(abpath)
        sum_tpr = 0
        s2_tpr = 0
        sum_fpr = 0
        s2_fpr = 0
        for i in range(NFOLDS):
            normal_counts_train = normal_counts[:i * fold_size] + normal_counts[(i + 1) * fold_size:]
            normal_counts_test = normal_counts[i*fold_size:(i+1)*fold_size]
            cs = confusion_matrices(normal_counts_train,
                                    normal_counts_test,
                                    abnormal_counts)
            tpr = cs[:, 1, 1] / (cs[:, 1, 0] + cs[:, 1, 1])
            fpr = cs[:, 0, 1] / (cs[:, 0, 0] + cs[:, 0, 1])
            auc = np.trapz(tpr, fpr)
            # print(tpr)
            sum_tpr += tpr
            s2_tpr += tpr * tpr
            sum_fpr += fpr
            s2_fpr += fpr * fpr
        mean_tpr = sum_tpr / NFOLDS
        std_tpr = np.sqrt(s2_tpr / NFOLDS - mean_tpr * mean_tpr) * NFOLDS / (NFOLDS - 1)
        mean_fpr = sum_fpr / NFOLDS
        std_fpr = np.sqrt(s2_fpr / NFOLDS - mean_fpr * mean_fpr) * NFOLDS / (NFOLDS - 1)
        print(f"{exp + '/' + abname:34s} | {mean_tpr[0]:.3f} | {std_tpr[0]:.3f}")
    print("false positive rate:")
    print(f"{exp + '/false positive':34s} | {mean_fpr[0]:.3f} | {std_fpr[0]:.3f}")