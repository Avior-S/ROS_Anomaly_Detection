import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve
import pandas as pd

# Consts
INF = -5
SUP = 50
TOTAL_ACC = 0.0
EXP_NUM = 0

AUROCs = []

columns = []


def load_directory(directory, file_prefix):
    runs = []
    for i in range(len(os.listdir(directory))):
        f = os.path.join(directory, file_prefix + str(i + 1) + ".csv")
        if os.path.isfile(f):
            runs.append(np.genfromtxt(f, delimiter=",", skip_header=1))
            columns.extend(np.genfromtxt(f, dtype=str, delimiter=",")[0])
    return runs, np.concatenate([np.reshape(r, -1) for r in runs])


def plot_hist(runs, title="TEST"):
    fig, axs = plt.subplots(1)
    for i, r in enumerate(runs):
        temp = [np.logical_and(r > INF, r < SUP)]
        axs.hist(np.reshape(r, -1), bins='auto', density=True)
    axs.set_title(title)
    axs.set_xlim((INF, SUP))
    axs.plot()


def ks_test(runs, more_runs=[]):
    ks_test = []
    if len(more_runs) == 0:  # Compare to itself
        for j, rj in enumerate(runs):
            other_runs = np.concatenate([np.reshape(runs[i], -1) for i in range(len(runs)) if i != j])
            or_reshape = np.reshape(other_runs, -1)
            rj_reshape = np.reshape(rj, -1)
            ks_test.append(ks_2samp(np.reshape(other_runs, -1), np.reshape(rj, -1)).statistic)
    else:  # Compare to other
        n = 2
        # more_runs - it's normal_normal runs
        # runs - it's potentialy anomaly runs
        for j, rj in enumerate(runs):
            rj = rj[len(rj) * (n - 2) // 3:]
            ks_test.append(ks_2samp(np.reshape(more_runs, -1), np.reshape(rj, -1)).statistic)
    return ks_test


def plot_ks_test_results(norm_norm, norm_anomaly, anomaly_anomaly, title=''):
    plt.title(title)
    plt.hist(norm_norm, bins="auto", density=True, alpha=0.5, label='normal vs normal')
    plt.hist(norm_anomaly, bins="auto", density=True, alpha=0.5, label='anomaly vs normal')
    plt.hist(anomaly_anomaly, bins="auto", density=True, alpha=0.5, label='normal vs normal test')
    plt.legend()
    plt.savefig(title + '.png')


def threshold_dist(norm_norm, threshold_percent=0.95):
    sorted_n_n = norm_norm.copy()
    sorted_n_n.sort(reverse=False)
    threshold_index = int(len(sorted_n_n) * threshold_percent)
    #     print(sorted_n_n, threshold_index)
    avg_dist = np.min(sorted_n_n[threshold_index:])
    print('Min distance of ' + str((threshold_percent * 100)) + '% confidence level:', avg_dist)
    return avg_dist


def detect_anomalies(ks_norm_anomaly, ks_norm_norm, ks_norm_test, threshold_percent=0.95, title="ROC", files_names=[]):
    global EXP_NUM
    global TOTAL_ACC
    EXP_NUM += 1
    anomaly_files_name = files_names[2]
    dict_exists = {}
    for file_name in anomaly_files_name:
        scenario = file_name.split('\\')[1]
        if scenario not in dict_exists.keys():
            dict_exists[scenario] = 0
        dict_exists[scenario] += 1
    avg_dist = threshold_dist(ks_norm_norm, threshold_percent)
    count_anom = [1 if i >= avg_dist else 0 for i in ks_norm_anomaly]
    count_anom_by_scenario = 0
    dict_detect = {}
    for scenario in dict_exists.keys():
        dict_detect[scenario] = count_anom[count_anom_by_scenario:count_anom_by_scenario + dict_exists[scenario]]
        count_anom_by_scenario += dict_exists[scenario]
        print("Scenario " + scenario + " true positive rate: " + str(sum(dict_detect[scenario])/dict_exists[scenario]))
    count_false_anomalies = [1 if i >= avg_dist else 0 for i in ks_norm_test]
    dict_detect['Normal (FP)'] = count_false_anomalies
    print('Normal (FP) ' + str(sum(count_false_anomalies)/len(count_false_anomalies)))
    TOTAL_ACC += sum(count_anom) * 100 / len(ks_norm_anomaly)
    print('Total anomalies detected:', sum(count_anom), 'of', len(ks_norm_anomaly),
          ', accuracy:' + str(sum(count_anom) * 100 / len(ks_norm_anomaly)) + '%')

    # The x-axis showing 1 – specificity (= false positive fraction = FP/(FP+TN))
    # The y-axis showing sensitivity (= true positive fraction = TP/(TP+FN))
    x = []
    y = []
    nn_copy = ks_norm_norm.copy()
    nn_copy.sort(reverse=True)
    tp = 0
    fp = 0
    for t in nn_copy:
        true_positive = sum([1 if i >= t else 0 for i in ks_norm_anomaly])
        true_negative = sum([1 if i < t else 0 for i in ks_norm_norm])
        false_positive = sum([1 if i >= t else 0 for i in ks_norm_norm])
        false_negative = sum([1 if i < t else 0 for i in ks_norm_anomaly])
        x.append(false_positive / (false_positive + true_negative))
        y.append(true_positive / (true_positive + false_negative))
        tp = tp + true_positive / len(ks_norm_anomaly)
        fp = fp + false_positive / len(ks_norm_norm)
        # print('True positive ', tp * 100/len(nn_copy))
        # print('True negative', fp * 100/len(nn_copy))
    np.savetxt('x_' + title + '.csv', x, delimiter=",")
    np.savetxt('y_' + title + '.csv', y, delimiter=",")
    fig, axs = plt.subplots(1)
    axs.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
    axs.plot(x, y, marker='o')
    axs.plot(axs.get_xlim(), axs.get_ylim(), ls="--", c=".3")
    axs.set_title(title)
    axs.set_ylabel('True Positive Rate')
    axs.set_xlabel('False Positive Rate')
    plt.show()
    fig.savefig(title + '.png')
    auroc = 0
    for i in range(len(x) - 1):
        auroc += ((x[i + 1] - x[i]) * (y[i + 1] + y[i])) / 2
    print('AUROC:', auroc)
    print('True Positive Rate', sum(count_anom) * 100 / len(ks_norm_anomaly))
    print('False Positive Rate: ', sum(count_false_anomalies) * 100 / len(ks_norm_test))
    return [title.replace("ROC", ''), auroc, tp * 100 / len(nn_copy),
            sum(count_false_anomalies) * 100 / len(ks_norm_test)], dict_detect

def main():
    TITLE_NORMAL = 'Real Panda'
    TITLE_ANOMALY = 'Real Panda Gripper attack'
    NORM_COUNTS_PATH = './data/real_panda/normal'
    NORM_FILE_PREFIX = 'counts_panda_normal'
    ANOMALY_COUNTS_PATH = './data/real_panda/abnormal/gripper_attack'
    ANOM_FILE_PREFIX = 'counts_panda_gripper_attack'

    #
    # TITLE_NORMAL = 'Real Panda'
    # TITLE_ANOMALY = 'Real Panda Change object weight'
    # NORM_COUNTS_PATH = './data/real_panda/normal'
    # NORM_FILE_PREFIX = 'counts_panda_normal'
    # ANOMALY_COUNTS_PATH = './data/real_panda/abnormal/change_obj_weight'
    # ANOM_FILE_PREFIX = 'counts_panda_change_wieght'

    runs, all_runs = load_directory(NORM_COUNTS_PATH, NORM_FILE_PREFIX)
    plot_hist(runs, TITLE_NORMAL)

    anomaly_runs, all_anomaly_runs = load_directory(ANOMALY_COUNTS_PATH, ANOM_FILE_PREFIX)
    plot_hist(anomaly_runs, TITLE_ANOMALY)

    ks_normal_normal = ks_test(runs)
    ks_normal_anomaly = ks_test(anomaly_runs, all_runs)
    ks_anomaly_anomaly = ks_test(anomaly_runs)
    plot_ks_test_results(ks_normal_normal, ks_normal_anomaly, ks_anomaly_anomaly, TITLE_ANOMALY)
    print('Anomaly Max dist:', max(ks_normal_anomaly), ' Anomaly Min dist: ', min(ks_normal_anomaly))
    auroc = detect_anomalies(ks_norm_anomaly=ks_normal_anomaly, ks_norm_norm=ks_normal_normal, title='ROC ' + TITLE_ANOMALY)
    AUROCs.append(auroc)


if __name__=='__main__':
    main()


"""
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve
import pandas as pd

# Consts
INF = -5
SUP = 50
TOTAL_ACC = 0.0
EXP_NUM = 0

AUROCs = []

columns = []


def load_directory(directory, file_prefix):
    runs = []
    for i in range(len(os.listdir(directory))):
        f = os.path.join(directory, file_prefix + str(i + 1) + ".csv")
        if os.path.isfile(f):
            runs.append(np.genfromtxt(f, delimiter=",", skip_header=1))
            columns.extend(np.genfromtxt(f, dtype=str, delimiter=",")[0])
    return runs, np.concatenate([np.reshape(r, -1) for r in runs])


def plot_hist(runs, title="TEST"):
    fig, axs = plt.subplots(1)
    for i, r in enumerate(runs):
        temp = [np.logical_and(r > INF, r < SUP)]
        axs.hist(np.reshape(r, -1), bins='auto', density=True)
    axs.set_title(title)
    axs.set_xlim((INF, SUP))
    axs.plot()


def ks_test(runs, more_runs=[]):
    ks_test = []
    if len(more_runs) == 0:  # Compare to itself
        for j, rj in enumerate(runs):
            other_runs = np.concatenate([np.reshape(runs[i], -1) for i in range(len(runs)) if i != j])

            or_reshape = np.reshape(other_runs, -1)
            or_len = len(or_reshape)
            or_probability_mass = np.array([list(or_reshape).count(x)/or_len for x in range(0,SUP)])
            or_cumulative_probability = []
            prob_sum = 0
            for i in or_probability_mass:
                prob_sum += i
                or_cumulative_probability.append(prob_sum)
            rj_reshape = np.reshape(rj, -1)
            rj_len = len(rj_reshape)
            rj_probability_mass = np.array([list(rj_reshape).count(x)/rj_len for x in range(0, SUP)])
            rj_cumulative_probability = []
            prob_sum = 0
            for i in rj_probability_mass:
                prob_sum += i
                rj_cumulative_probability.append(prob_sum)
            ks_test.append(ks_2samp(or_cumulative_probability, rj_cumulative_probability).statistic)
            # plt.plot(range(0,SUP), rj_cumulative_probability, 'r', range(0,SUP), or_cumulative_probability, 'b')
            # plt.show()
            # plot_hist([or_count,rj_count], str(j))
    else:  # Compare to other
        # more_runs - it's normal_normal runs
        # runs - it's potentially anomaly runs
        for j, rj in enumerate(runs):
            mr_reshape = np.reshape(more_runs, -1)
            mr_len = len(mr_reshape)
            mr_probability_mass = np.array([list(mr_reshape).count(x)/mr_len for x in range(0,SUP)])
            mr_cumulative_probability = []
            prob_sum = 0
            for i in mr_probability_mass:
                prob_sum += i
                mr_cumulative_probability.append(prob_sum)
            rj_reshape = np.reshape(rj, -1)
            rj_len = len(rj_reshape)
            rj_probability_mass = np.array([list(rj_reshape).count(x)/rj_len for x in range(0, SUP)])
            rj_cumulative_probability = []
            prob_sum = 0
            for i in rj_probability_mass:
                prob_sum += i
                rj_cumulative_probability.append(prob_sum)
            ks_test.append(ks_2samp(mr_cumulative_probability, rj_cumulative_probability).statistic)
    return ks_test


def plot_ks_test_results(norm_norm, norm_anomaly, anomaly_anomaly, title=''):
    plt.title(title)
    plt.hist(norm_norm, bins="auto", density=True, alpha=0.5, label='normal vs normal')
    plt.hist(norm_anomaly, bins="auto", density=True, alpha=0.5, label='anomaly vs normal')
    plt.hist(anomaly_anomaly, bins="auto", density=True, alpha=0.5, label='normal vs normal test')
    plt.legend()
    plt.savefig(title + '.png')


def threshold_dist(norm_norm, threshold_percent=0.95):
    sorted_n_n = norm_norm.copy()
    sorted_n_n.sort(reverse=False)
    threshold_index = int(len(sorted_n_n) * threshold_percent)
    #     print(sorted_n_n, threshold_index)
    avg_dist = np.min(sorted_n_n[threshold_index:])
    print('Min distance of ' + str((threshold_percent * 100)) + '% confidence level:', avg_dist)
    return avg_dist


def detect_anomalies(ks_norm_anomaly, ks_norm_norm, ks_norm_test, threshold_percent=0.95, title="ROC", files_names=[]):
    global EXP_NUM
    global TOTAL_ACC
    EXP_NUM += 1
    anomaly_files_name = files_names[2]
    dict_exists = {}
    for file_name in anomaly_files_name:
        scenario = file_name.split('\\')[1]
        if scenario not in dict_exists.keys():
            dict_exists[scenario] = 0
        dict_exists[scenario] += 1
    avg_dist = threshold_dist(ks_norm_norm, threshold_percent)
    count_anom = [1 if i >= avg_dist else 0 for i in ks_norm_anomaly]
    count_anom_by_scenario = 0
    dict_detect = {}
    for scenario in dict_exists.keys():
        dict_detect[scenario] = count_anom[count_anom_by_scenario:count_anom_by_scenario + dict_exists[scenario]]
        count_anom_by_scenario += dict_exists[scenario]
        print("Scenario " + scenario + " true positive rate: " + str(sum(dict_detect[scenario])/dict_exists[scenario]))
    count_false_anomalies = [1 if i >= avg_dist else 0 for i in ks_norm_test]
    dict_detect['Normal (FP)'] = count_false_anomalies
    print('Normal (FP) ' + str(sum(count_false_anomalies)/len(count_false_anomalies)))
    TOTAL_ACC += sum(count_anom) * 100 / len(ks_norm_anomaly)
    print('Total anomalies detected:', sum(count_anom), 'of', len(ks_norm_anomaly),
          ', accuracy:' + str(sum(count_anom) * 100 / len(ks_norm_anomaly)) + '%')

    # The x-axis showing 1 – specificity (= false positive fraction = FP/(FP+TN))
    # The y-axis showing sensitivity (= true positive fraction = TP/(TP+FN))
    x = []
    y = []
    nn_copy = ks_norm_norm.copy()
    nn_copy.sort(reverse=True)
    tp = 0
    fp = 0
    for t in nn_copy:
        true_positive = sum([1 if i >= t else 0 for i in ks_norm_anomaly])
        true_negative = sum([1 if i < t else 0 for i in ks_norm_norm])
        false_positive = sum([1 if i >= t else 0 for i in ks_norm_norm])
        false_negative = sum([1 if i < t else 0 for i in ks_norm_anomaly])
        x.append(false_positive / (false_positive + true_negative))
        y.append(true_positive / (true_positive + false_negative))
        tp = tp + true_positive / len(ks_norm_anomaly)
        fp = fp + false_positive / len(ks_norm_norm)
        # print('True positive ', tp * 100/len(nn_copy))
        # print('True negative', fp * 100/len(nn_copy))
    np.savetxt('x_' + title + '.csv', x, delimiter=",")
    np.savetxt('y_' + title + '.csv', y, delimiter=",")
    fig, axs = plt.subplots(1)
    axs.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
    axs.plot(x, y, marker='o')
    axs.plot(axs.get_xlim(), axs.get_ylim(), ls="--", c=".3")
    axs.set_title(title)
    axs.set_ylabel('True Positive Rate')
    axs.set_xlabel('False Positive Rate')
    plt.show()
    fig.savefig(title + '.png')
    auroc = 0
    for i in range(len(x) - 1):
        auroc += ((x[i + 1] - x[i]) * (y[i + 1] + y[i])) / 2
    print('AUROC:', auroc)
    print('True Positive Rate', sum(count_anom) * 100 / len(ks_norm_anomaly))
    print('False Positive Rate: ', sum(count_false_anomalies) * 100 / len(ks_norm_test))
    return [title.replace("ROC", ''), auroc, tp * 100 / len(nn_copy),
            sum(count_false_anomalies) * 100 / len(ks_norm_test)], dict_detect

def main():
    TITLE_NORMAL = 'Real Panda'
    TITLE_ANOMALY = 'Real Panda Gripper attack'
    NORM_COUNTS_PATH = './data/real_panda/normal'
    NORM_FILE_PREFIX = 'counts_panda_normal'
    ANOMALY_COUNTS_PATH = './data/real_panda/abnormal/gripper_attack'
    ANOM_FILE_PREFIX = 'counts_panda_gripper_attack'

    #
    # TITLE_NORMAL = 'Real Panda'
    # TITLE_ANOMALY = 'Real Panda Change object weight'
    # NORM_COUNTS_PATH = './data/real_panda/normal'
    # NORM_FILE_PREFIX = 'counts_panda_normal'
    # ANOMALY_COUNTS_PATH = './data/real_panda/abnormal/change_obj_weight'
    # ANOM_FILE_PREFIX = 'counts_panda_change_wieght'

    runs, all_runs = load_directory(NORM_COUNTS_PATH, NORM_FILE_PREFIX)
    plot_hist(runs, TITLE_NORMAL)

    anomaly_runs, all_anomaly_runs = load_directory(ANOMALY_COUNTS_PATH, ANOM_FILE_PREFIX)
    plot_hist(anomaly_runs, TITLE_ANOMALY)

    ks_normal_normal = ks_test(runs)
    ks_normal_anomaly = ks_test(anomaly_runs, all_runs)
    ks_anomaly_anomaly = ks_test(anomaly_runs)
    plot_ks_test_results(ks_normal_normal, ks_normal_anomaly, ks_anomaly_anomaly, TITLE_ANOMALY)
    print('Anomaly Max dist:', max(ks_normal_anomaly), ' Anomaly Min dist: ', min(ks_normal_anomaly))
    auroc = detect_anomalies(ks_norm_anomaly=ks_normal_anomaly, ks_norm_norm=ks_normal_normal, title='ROC ' + TITLE_ANOMALY)
    AUROCs.append(auroc)


if __name__=='__main__':
    main()
"""