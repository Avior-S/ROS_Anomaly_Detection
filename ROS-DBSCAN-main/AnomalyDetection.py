import os

import Dataset as DS
import Dataset2 as DS2
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from Style import Configure as Conf
import Measurements as M
import numpy as np
from scipy.spatial.distance import cdist
from operator import itemgetter
# from Isolation_Forest_detection import iso_forest
import re
import Confidence_Intervals as CI


def removes_nan(datasets):
    new_datasets = []
    for dataset in datasets:
        new_datasets.append(list(map(lambda df: df.fillna(0), dataset)))
    return new_datasets


def normalization(datasets):
    trainings, positives, negatives = datasets
    t_df = pd.concat(trainings, ignore_index=True)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(t_df)
    n_trainings = list(map(lambda df: pd.DataFrame(min_max_scaler.transform(df), columns=t_df.columns), trainings))
    n_positives = list(map(lambda df: pd.DataFrame(min_max_scaler.transform(df), columns=t_df.columns), positives))
    n_negatives = list(map(lambda df: pd.DataFrame(min_max_scaler.transform(df), columns=t_df.columns), negatives))
    return n_trainings, n_positives, n_negatives


# The important features are the ones that influence more the components and thus,
# have a large absolute value/score on the component.
def most_influence_feature_by_pca(datasets, n=25):
    trainings, positives, negatives = datasets
    # t_df = pd.concat(trainings, ignore_index=True)
    initial_feature_names = trainings[0].columns
    dic = dict.fromkeys(range(0, trainings[0].shape[1]), 0)
    for training in trainings:
        pca = PCA(n_components=n)
        pca = pca.fit(training)
        for arr in pca.components_:
            abs_arr = np.abs(arr)
            for i in range(len(abs_arr)):
                dic[i] += abs_arr[i]
    dic_most_important_feature = dict(sorted(dic.items(), key=itemgetter(1), reverse=True)[:n])
    most_important_index = list(dic_most_important_feature.keys())
    # get the names
    most_important_names = [initial_feature_names[most_important_index[i]] for i in range(n)]
    # ipca_training = list(map(lambda df: pca.transform(df), trainings))
    # ipca_positive = list(map(lambda df: pca.transform(df), positives))
    # ipca_negative = list(map(lambda df: pca.transform(df), negatives))
    return most_important_names


def topics_filter(datasets, topics=None):
    if topics is None:
        topics = []
    trainings, positives, negatives = datasets
    flt_training = list(map(lambda df: df[topics], trainings))
    flt_positive = list(map(lambda df: df[topics], positives))
    flt_negative = list(map(lambda df: df[topics], negatives))
    return flt_training, flt_positive, flt_negative


def get_mic_topics(exp):
    if "panda" in exp:
        topics = []
        for epv in ['effort', 'position', 'velocity']:
            for i in range(1, 8):
                topics.append('panda_joint' + str(i) + ' ' + epv)
            for i in range(1, 3):
                topics.append('panda_finger_joint' + str(i) + ' ' + epv)
    else:
        topics = ['linear velocity x', 'linear velocity y', 'angular velocity z']
    return topics


def find_max_longest_sequence(trainings_names, trainings_preds, alpha=0.95):
    longests = []
    for f, y_pred in zip(trainings_names, trainings_preds):
        longest = M.Measurements.longest_sequence(y_pred, Conf.NEGATIVE_LABEL)
        longests.append(longest)
    longests.sort()
    max_longest = longests[int(round(len(longests) * alpha)) - 1]
    ## print(longests)
    ## print("max:    " + str(max_longest))
    return max_longest


def predictions_information(datasets_preds):
    def get_acc_info(title, paths_preds, pred_value, max_longest):
        # global an
        ans = "------------------------ %s ------------------------\n" % (title,)
        tmp = ""
        longests = 0
        anomalies = 0
        warnings = 0
        faulty_runs = 0
        acc_size = 5
        f_size = 30
        count = 0
        for f, y_pred in paths_preds:
            count += 1
            y_true = [pred_value] * len(y_pred)
            acc = round(M.Measurements.accuracy_score(y_true, y_pred), acc_size)
            anomalies += len(list(filter(lambda x: x != pred_value, y_pred)))
            longest = M.Measurements.longest_sequence(y_pred, Conf.NEGATIVE_LABEL)
            longests += longest
            warning = M.Measurements.count_warnings(y_pred, Conf.NEGATIVE_LABEL, max_longest + 1)
            if warning > 0:
                faulty_runs += 1
            warnings += warning
            tmp += "%s%s\taccuracy: %s%s\tlongest negative: %s\twarnings = %s\n" % (
            f, " " * (f_size - (len(f) - 2)), acc, " " * (acc_size - (len(str(acc)) - 2)), longest, warning)
            # M.Measurements.draw_chart(f, options.charts + f[:-4],y_pred,Conf.NEGATIVE_LABEL)
        # an = longests/len(paths_preds)
        ans = ans + "longest average = %s, anomalies = %s,  total warnings = %s,  faulty runs = %s,  non-faulty runs = %s\n" % (
        longests / count, anomalies, warnings, faulty_runs, count - faulty_runs)
        return ans + tmp

    trainings_preds, positives_preds, negatives_preds = datasets_preds.get_predictions()
    trainings_names, positives_names, negatives_names = datasets_preds.get_names()
    ans = ""
    max_longest = find_max_longest_sequence(trainings_names, trainings_preds)
    ## print("max threshold " + str(max_longest))
    ans += get_acc_info("training sets", zip(trainings_names, trainings_preds), Conf.POSITIVE_LABEL, max_longest) + "\n"
    ans += get_acc_info("positive sets", zip(positives_names, positives_preds), Conf.POSITIVE_LABEL, max_longest) + "\n"
    ans += get_acc_info("negative sets", zip(negatives_names, negatives_preds), Conf.POSITIVE_LABEL, max_longest) + "\n"
    return ans


def dict_sum_preds(multi_paths_preds, max_longest, dict_preds={}):
    for paths_preds in multi_paths_preds:
        for f, y_pred in paths_preds:
            if f not in dict_preds:
                dict_preds[f] = 0
            warning = M.Measurements.count_warnings(y_pred, Conf.NEGATIVE_LABEL, max_longest + 1)
            if warning > 0:
                dict_preds[f] += 1
    return dict_preds


def sum_result(dict):
    tmp = "not exist file"
    count = 0
    anomaly_count = 0
    lst = []
    for key in dict.keys():
        if tmp not in key:
            if not count == 0:
                if 'type' in tmp:
                    tmp = 'normal '
                ## print(tmp)
                ## print("number of runs detected as anomaly: " + str(anomaly_count) + " from " + str(count))
                lst.append(anomaly_count/count)
            count = 0
            anomaly_count = 0
            tmp = key.split('\\')[1]
        count += 1
        if dict[key] > 0:
            anomaly_count += 1
    if 'type' in tmp:
        tmp = 'normal '
    ## print(tmp)
    ## print("number of runs detected as anomaly: " + str(anomaly_count) + " from " + str(count))
    lst.append(anomaly_count / count)
    return lst

def extract_group_names(data):
    """ Extracts unique group names from dictionary keys while preserving order """
    group_names = []
    seen = set()
    for key in data.keys():
        match = re.search(r'\\([^\\]+)\\[^\\]+\.csv$', key)
        if match:
            group_name = match.group(1)
            if group_name == "type1":
                group_name = "normal"
            if group_name not in seen:
                seen.add(group_name)
                group_names.append(group_name)
    return group_names

def my_DBSCAN(d, scenario):
    from ROS_DBSCAN import run_dbscan_n_predict
    df = pd.DataFrame()
    similar_columns = d.find_similar_columns_in_training()
    d.filter_by_columns(similar_columns)
    mic_topics = get_mic_topics(scenario)
    mic_dfs = d.get_copied_datasets()
    flt_mic_dfs = topics_filter(mic_dfs, mic_topics)
    norm_flt_mic_dfs = normalization(flt_mic_dfs)
    dfs, p_dfs = run_dbscan_n_predict(norm_flt_mic_dfs)
    d.set_predictions(p_dfs[0], p_dfs[1], p_dfs[2])
    # ## print(predictions_information(d))
    trainings_preds, positives_preds, negatives_preds = d.get_predictions()
    dfs[0][0].to_excel('example_df.xlsx', index=False)
    trainings_names, positives_names, negatives_names = d.get_names()
    multi_paths_preds = [zip(positives_names, positives_preds), zip(negatives_names, negatives_preds)]
    max_longest = find_max_longest_sequence(trainings_names, trainings_preds)
    mic_dict_preds = dict_sum_preds(multi_paths_preds, max_longest, dict_preds={})
    ## print("summarize Result for Mic features:")
    df['mic'] = sum_result(mic_dict_preds)

    # Macro Anomaly Detection
    ## print("\n\n ------------------------------ PCA ------------------------------------ \n\n")
    mac_dfs = d.get_copied_datasets()
    d_mac_dfs = DS.drop_topics(mac_dfs, mic_topics, 'Active')
    n_d_mac_dfs = normalization(d_mac_dfs)
    mac_topics = most_influence_feature_by_pca(n_d_mac_dfs, n=27)
    flt_n_d_mac_dfs = topics_filter(n_d_mac_dfs, mac_topics)
    dfs, p_dfs = run_dbscan_n_predict(flt_n_d_mac_dfs)
    d.set_predictions(p_dfs[0], p_dfs[1], p_dfs[2])
    # ## print(predictions_information(d))
    trainings_preds, positives_preds, negatives_preds = d.get_predictions()
    trainings_names, positives_names, negatives_names = d.get_names()
    multi_paths_preds = [zip(positives_names, positives_preds), zip(negatives_names, negatives_preds)]
    max_longest = find_max_longest_sequence(trainings_names, trainings_preds)
    mac_dict_preds = dict_sum_preds(multi_paths_preds, max_longest, {})
    # print("summarize Result for Macro features:")
    # filtered_mac_dict_preds = {k: v for k, v in mac_dict_preds.items() if k.startswith("") and v == 1}
    # print(filtered_mac_dict_preds)
    df['mac'] = sum_result(mac_dict_preds)

    for key in mic_dict_preds.keys():
        mic_dict_preds[key] += mac_dict_preds[key]

    ## print("\n\nUnion Result:")
    df['union'] = sum_result(mic_dict_preds)
    group_names = extract_group_names(mic_dict_preds)
    df.index = group_names
    # df.to_excel('C:\\Users\Avior\PycharmProjects\ROS-DBSCAN\\result.xlsx')
    return df


def get_mic_df(d, scenario):
    similar_columns = d.find_similar_columns_in_training()
    d.filter_by_columns(similar_columns)
    mic_topics = get_mic_topics(scenario)
    mic_dfs = d.get_copied_datasets()
    flt_mic_dfs = topics_filter(mic_dfs, mic_topics)
    return normalization(flt_mic_dfs)


def get_mac_df(d, scenario):
    mic_topics = get_mic_topics(scenario)
    mac_dfs = d.get_copied_datasets()
    d_mac_dfs = DS.drop_topics(mac_dfs, mic_topics, 'Active')
    n_d_mac_dfs = normalization(d_mac_dfs)
    mac_topics = most_influence_feature_by_pca(n_d_mac_dfs, n=27)
    flt_n_d_mac_dfs = topics_filter(n_d_mac_dfs, mac_topics)
    return flt_n_d_mac_dfs


def my_AEAD(d, scenario):
    import ROS_AEAD as AEAD
    df = pd.DataFrame()
    similar_columns = d.find_similar_columns_in_training()
    d.filter_by_columns(similar_columns)
    mic_topics = get_mic_topics(scenario)
    mic_dfs = d.get_copied_datasets()
    flt_mic_dfs = topics_filter(mic_dfs, mic_topics)
    norm_flt_mic_dfs = normalization(flt_mic_dfs)
    if "panda" in scenario:
        feat = 27
    else:
        feat = 3
    dfs, n_dfs = AEAD.AEAD(norm_flt_mic_dfs, feat=feat) # In panda scenario feat = 27, in turtlebot3 feat = 3
    d.set_predictions(n_dfs[0], n_dfs[1], n_dfs[2])
    ## print(predictions_information(d))
    trainings_preds, positives_preds, negatives_preds = d.get_predictions()
    trainings_names, positives_names, negatives_names = d.get_names()
    multi_paths_preds = [zip(positives_names, positives_preds), zip(negatives_names, negatives_preds)]
    max_longest = find_max_longest_sequence(trainings_names, trainings_preds)
    mic_dict_preds = dict_sum_preds(multi_paths_preds, max_longest, dict_preds={})
    ## print("summarize Result for Mic features:")
    df['mic'] = sum_result(mic_dict_preds)

    # Macro Anomaly Detection
    ## print("\n\n ------------------------------ PCA ------------------------------------ \n\n")
    mac_dfs = d.get_copied_datasets()
    d_mac_dfs = DS.drop_topics(mac_dfs, mic_topics, 'Active')
    n_d_mac_dfs = normalization(d_mac_dfs)
    mac_topics = most_influence_feature_by_pca(n_d_mac_dfs, n=27)
    flt_n_d_mac_dfs = topics_filter(n_d_mac_dfs, mac_topics)
    dfs, n_dfs = AEAD.AEAD(flt_n_d_mac_dfs)
    d.set_predictions(n_dfs[0], n_dfs[1], n_dfs[2])
    ## print(predictions_information(d))
    trainings_preds, positives_preds, negatives_preds = d.get_predictions()
    trainings_names, positives_names, negatives_names = d.get_names()
    multi_paths_preds = [zip(positives_names, positives_preds), zip(negatives_names, negatives_preds)]
    # multi_paths_preds = [zip(trainings_names, trainings_preds), zip(positives_names, positives_preds),
    #                      zip(negatives_names, negatives_preds)]
    max_longest = find_max_longest_sequence(trainings_names, trainings_preds)
    mac_dict_preds = dict_sum_preds(multi_paths_preds, max_longest, {})
    ## print("summarize Result for Macro features:")
    df['mac'] = sum_result(mac_dict_preds)

    for key in mic_dict_preds.keys():
        mic_dict_preds[key] += mac_dict_preds[key]

    ## print("\n\nUnion Result:")
    df['union'] = sum_result(mic_dict_preds)
    df.to_excel('result.xlsx')
    return df['union']


def my_ks(d, scenario):
    import kolmogorov_smirnov_old as ks
    # runs, all_runs = ks.load_directory(NORM_COUNTS_PATH, NORM_FILE_PREFIX)
    norm_mic_df = get_mic_df(d, scenario)
    runs = norm_mic_df
    normal_runs = [r.to_numpy() for r in runs[0]]
    all_normal_runs = np.concatenate([np.reshape(r, -1) for r in normal_runs])
    normal_test_runs = [r.to_numpy() for r in runs[1]]
    anomaly_runs = [r.to_numpy() for r in runs[2]]
    all_anomaly_runs = np.concatenate([np.reshape(r, -1) for r in anomaly_runs])
    TITLE = 'Normal'
    # ks.plot_hist(normal_runs, TITLE)

    ks_normal_normal = ks.ks_test(normal_runs)
    ks_normal_anomaly = ks.ks_test(anomaly_runs, all_normal_runs)
    ks_normal_test = ks.ks_test(normal_test_runs, all_normal_runs)
    TITLE_ANOMALY = 'MIX'
    files_names = d.get_names()
    ks.plot_ks_test_results(ks_normal_normal, ks_normal_anomaly, ks_normal_test, TITLE_ANOMALY)
    ## print('Anomaly Max dist:', max(ks_normal_anomaly), ' Anomaly Min dist: ', min(ks_normal_anomaly))
    auroc, dict_results_mic = ks.detect_anomalies(ks_norm_anomaly=ks_normal_anomaly, ks_norm_norm=ks_normal_normal,
                             ks_norm_test=ks_normal_test ,title='ROC ' + TITLE_ANOMALY, files_names=files_names)
    ks.AUROCs.append(auroc)
    ## print("-----------------      PCA       ------------------------")
    similar_columns = d.find_similar_columns_in_training()
    d.filter_by_columns(similar_columns)
    mac_df = get_mac_df(d, scenario)
    runs = mac_df
    normal_runs = [r.to_numpy() for r in runs[0]]
    temp = [np.reshape(r, -1) for r in normal_runs]
    all_normal_runs = np.concatenate(temp)
    normal_test_runs = [r.to_numpy() for r in runs[1]]
    anomaly_runs = [r.to_numpy() for r in runs[2]]
    all_anomaly_runs = np.concatenate([np.reshape(r, -1) for r in anomaly_runs])
    TITLE = 'Normal'
    # ks.plot_hist(normal_runs, TITLE)

    ks_normal_normal = ks.ks_test(normal_runs)
    ks_normal_anomaly = ks.ks_test(anomaly_runs, all_normal_runs)
    ks_normal_test = ks.ks_test(normal_test_runs, all_normal_runs)
    TITLE_ANOMALY = 'MIX'
    files_names = d.get_names()
    ks.plot_ks_test_results(ks_normal_normal, ks_normal_anomaly, ks_normal_test, TITLE_ANOMALY)
    ## print('Anomaly Max dist:', max(ks_normal_anomaly), ' Anomaly Min dist: ', min(ks_normal_anomaly))
    auroc, dict_results_mac = ks.detect_anomalies(ks_norm_anomaly=ks_normal_anomaly, ks_norm_norm=ks_normal_normal,
                                ks_norm_test=ks_normal_test, title='ROC ' + TITLE_ANOMALY, files_names=files_names)
    ks.AUROCs.append(auroc)
    dict_final_results = {}
    for key in dict_results_mac:
        union_result = [dict_results_mic[key][i] or dict_results_mac[key][i] for i in range(0, len(dict_results_mic[key]))]
        key_len = len(dict_results_mic[key])
        dict_final_results[key] = [sum(dict_results_mic[key])/key_len, sum(dict_results_mac[key])/key_len, sum(union_result)/key_len]
    df = pd.DataFrame.from_dict(dict_final_results, orient='index', columns=['mic', 'mac', 'Union'])
    df.to_excel('ks_result.xlsx')

def david_ks(d):
    import kolmogorov_smirnov as ks
    # runs, all_runs = ks.load_directory(NORM_COUNTS_PATH, NORM_FILE_PREFIX)
    count_columns = d.find_columns_contains("Count")
    d.filter_by_columns(count_columns)
    runs = d.get_copied_datasets()
    normal_runs = [r.to_numpy() for r in runs[0]]
    all_normal_runs = np.concatenate([np.reshape(r, -1) for r in normal_runs])
    normal_test_runs = [r.to_numpy() for r in runs[1]]
    anomaly_runs = [r.to_numpy() for r in runs[2]]
    all_anomaly_runs = np.concatenate([np.reshape(r, -1) for r in anomaly_runs])
    TITLE = 'Normal'
    # ks.plot_hist(normal_runs, TITLE)

    ks_normal_normal = ks.ks_test(normal_runs)
    ks_normal_anomaly = ks.ks_test(anomaly_runs, all_normal_runs)
    ks_normal_test = ks.ks_test(normal_test_runs, all_normal_runs)
    TITLE_ANOMALY = 'MIX'
    files_names = d.get_names()
    ks.plot_ks_test_results(ks_normal_normal, ks_normal_anomaly, ks_normal_test, TITLE_ANOMALY)
    ## print('Anomaly Max dist:', max(ks_normal_anomaly), ' Anomaly Min dist: ', min(ks_normal_anomaly))
    auroc, dict_results = ks.detect_anomalies(ks_norm_anomaly=ks_normal_anomaly, ks_norm_norm=ks_normal_normal,
                                                  ks_norm_test=ks_normal_test, threshold_percent=0.95, title='ROC ' + TITLE_ANOMALY,
                                                  files_names=files_names)
    ks.AUROCs.append(auroc)


def save_with_title(df, file_path, title):
    # Check if the file exists
    file_exists = os.path.exists(file_path)

    # Open file in append mode
    with open(file_path, mode='a') as file:
        # Write the title before the DataFrame
        if not file_exists:
            # If file doesn't exist, write the title and header for the first DataFrame
            file.write(f"{title}\n")
            df.to_csv(file, mode='a', header=True, index=False)
        else:
            # For subsequent DataFrames, just write the title before appending the data
            file.write(f"{title}\n")
            df.to_csv(file, mode='a', header=False, index=False)



scenarios = ['real_panda', 'sim_panda', 'real_turtlebot3', 'sim_turtlebot3']
def main():

    for scenario in scenarios:
        print("--------------------------   " + scenario + "   ---------------------------")
        all_result = []
        NFOLDS = 10
        for i in range(NFOLDS):
            ## print("-----------------------------------------------------------------------")
            ## print("----------------------------   i = "+str(i)+"   --------------------------------")
            ## print("-----------------------------------------------------------------------")
            d = DS2.Datasets("data/"+scenario+"/normal/", "data/"+scenario+"/abnormal/", test_size=0.0, nfolds=NFOLDS, i=i)
            result = my_DBSCAN(d, scenario)
            # result = my_AEAD(d, scenario)
            # result = iso_forest(d, scenario)
            all_result.append(result)
            temp_result = result.copy()
            if i == 0:
                s_result = temp_result
                s2_result = temp_result*temp_result
            else:
                s_result += temp_result
                s2_result += temp_result*temp_result
        avg = s_result / NFOLDS
        std = np.sqrt(s2_result / NFOLDS - avg * avg) * NFOLDS / (NFOLDS - 1)
        # Assuming df_avg contains the averages and df_std contains the standard deviations
        df_combined = avg.astype(str) + " (" + std.astype(str) + ")"
        print("scenario     avg(std)")
        print(df_combined['union'])
        print()


        print ("Confidence Interval (16.5%-83.5%):")
        # חילוץ עמודת "union" לכל תרחיש בנפרד
        scenario_conf_intervals = []
        all_scenario_values = []
        scenario_labels = all_result[0].index.tolist()
        for i in range(len(all_result[0])):  # מספר התרחישים
            scenario_values = np.array([df.iloc[i]['union'] for df in all_result])
            all_scenario_values.append(scenario_values)
            ci_lower, median, ci_upper = CI.compute_confidence_interval(scenario_values)
            scenario_conf_intervals.append([ci_lower, median, ci_upper])
            print(f' {scenario_labels[i]}:  [{ci_lower:.4f}, {ci_upper:.4f}], Median: {median:.4f}')

        # הצגת הנתונים גרפית
        CI.plot_confidence_intervals(scenario_conf_intervals, scenario_labels, scenario)
        # CI.plot_all_confidence_intervals(scenario_conf_intervals, scenario_labels, all_scenario_values, scenario)


        # Iterating over DataFrames and saving them to CSV
        file_path = 'output.csv'
        # Delete the file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        # save_with_title(avg, file_path, f"DataFrame avg {scenario}")
        # save_with_title(std, file_path, f"DataFrame std {scenario}")
        # print("average:")
        # print(avg)
        # print("std:")
        # print(std)

if __name__=="__main__":
    main()
#my_AEAD(d)
# david_ks(d)
