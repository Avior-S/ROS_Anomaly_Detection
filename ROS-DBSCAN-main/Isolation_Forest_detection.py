import os

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import Dataset as DS
from AnomalyDetection import get_mic_topics, topics_filter, find_max_longest_sequence, dict_sum_preds, sum_result, \
    most_influence_feature_by_pca, normalization, extract_group_names
import pandas as pd
import Dataset2 as DS2
from sklearn.decomposition import PCA
from ROS_DBSCAN import convert_concat_to_datasets_n_prediction
from Style import Configure as Conf
Conf.NEGATIVE_LABEL = -1

def concat_datasets(datasets):
    trainings, positives, negatives = datasets
    training = pd.concat(trainings, ignore_index=True)
    positive = pd.concat(positives, ignore_index=True)
    negative = pd.concat(negatives, ignore_index=True)
    x_train = training
    x_test = pd.concat([positive, negative], ignore_index=True)
    x_dfs = pd.concat([training, positive, negative], ignore_index=True)
    return x_train, positive, negative, x_dfs


def iso_forest(d, scenario):
    similar_columns = d.find_similar_columns_in_training()
    d.filter_by_columns(similar_columns)
    mic_topics = get_mic_topics(scenario)
    mic_dfs = d.get_copied_datasets()
    flt_mic_dfs = topics_filter(mic_dfs, mic_topics)
    X_train, X_pos, X_neg, X_dfs = concat_datasets(flt_mic_dfs)

    # Scaling the data (important when features have different scales)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pos_scaled = scaler.transform(X_pos)
    X_neg_scaled = scaler.transform(X_neg)

    # Initialize Isolation Forest
    iso_forest = IsolationForest()

    # Fit the model on the scaled training data
    iso_forest.fit(X_train_scaled)

    # Predict anomalies in the test set
    trn_pred = iso_forest.predict(X_train_scaled)

    # Predict anomalies in the test set
    pos_pred = iso_forest.predict(X_pos_scaled)

    # Predict anomalies in the test set
    neg_pred = iso_forest.predict(X_neg_scaled)

    all_preds = np.concatenate((trn_pred, pos_pred, neg_pred))
    dfs, p_dfs = convert_concat_to_datasets_n_prediction(X_dfs, all_preds, flt_mic_dfs)
    d.set_predictions(p_dfs[0], p_dfs[1], p_dfs[2])
    # ## print(predictions_information(d))
    trainings_preds, positives_preds, negatives_preds = d.get_predictions()
    # dfs[0][0].to_excel('example_df.xlsx', index=False)
    trainings_names, positives_names, negatives_names = d.get_names()
    multi_paths_preds = [zip(positives_names, positives_preds), zip(negatives_names, negatives_preds)]
    max_longest = find_max_longest_sequence(trainings_names, trainings_preds)
    mic_dict_preds = dict_sum_preds(multi_paths_preds, max_longest, dict_preds={})
    df['mic'] = sum_result(mic_dict_preds)

    # ---------------------    MAC    ---------------------------------

    mac_dfs = d.get_copied_datasets()
    d_mac_dfs = DS.drop_topics(mac_dfs, mic_topics, 'Active')
    # Scaling the data (important when features have different scales)
    n_d_mac_dfs = normalization(d_mac_dfs)
    mac_topics = most_influence_feature_by_pca(n_d_mac_dfs, n=27)
    flt_n_d_mac_dfs = topics_filter(n_d_mac_dfs, mac_topics)

    X_train, X_pos, X_neg, X_dfs = concat_datasets(flt_n_d_mac_dfs)
    # Initialize Isolation Forest
    iso_forest = IsolationForest()

    # Fit the model on the scaled training data
    iso_forest.fit(X_train_scaled)

    # Predict anomalies in the test set
    trn_pred = iso_forest.predict(X_train_scaled)

    # Predict anomalies in the test set
    pos_pred = iso_forest.predict(X_pos_scaled)

    # Predict anomalies in the test set
    neg_pred = iso_forest.predict(X_neg_scaled)

    all_preds = np.concatenate((trn_pred, pos_pred, neg_pred))
    dfs, p_dfs = convert_concat_to_datasets_n_prediction(X_dfs, all_preds, flt_mic_dfs)
    d.set_predictions(p_dfs[0], p_dfs[1], p_dfs[2])
    # ## print(predictions_information(d))
    trainings_preds, positives_preds, negatives_preds = d.get_predictions()
    trainings_names, positives_names, negatives_names = d.get_names()
    multi_paths_preds = [zip(positives_names, positives_preds), zip(negatives_names, negatives_preds)]
    max_longest = find_max_longest_sequence(trainings_names, trainings_preds)
    mac_dict_preds = dict_sum_preds(multi_paths_preds, max_longest, {})
    ## print("summarize Result for Macro features:")
    print("summarize Result for Macro features:")
    filtered_mac_dict_preds = {k: v for k, v in mac_dict_preds.items() if k.startswith("") and v == 1}
    print(filtered_mac_dict_preds)
    df['mac'] = sum_result(mac_dict_preds)

    for key in mic_dict_preds.keys():
        mic_dict_preds[key] += mac_dict_preds[key]

    ## print("\n\nUnion Result:")
    df['union'] = sum_result(mic_dict_preds)
    return df

# Function to save DataFrame with title
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

file_path = 'output.csv'
# Delete the file if it exists
if os.path.exists(file_path):
    os.remove(file_path)

scenarios = ['real_panda', 'sim_panda', 'real_turtlebot3', 'sim_turtlebot3']
for scenario in scenarios:
    print("--------------------------   " + scenario + "   ---------------------------")
    NFOLDS = 10
    for i in range(NFOLDS):
        df = pd.DataFrame()
        ## print("-----------------------------------------------------------------------")
        ## print("----------------------------   i = "+str(i)+"   --------------------------------")
        ## print("-----------------------------------------------------------------------")
        d = DS2.Datasets("data/"+scenario+"/normal/", "data/"+scenario+"/abnormal/", test_size=0.0, nfolds=NFOLDS, i=i)
        similar_columns = d.find_similar_columns_in_training()
        d.filter_by_columns(similar_columns)
        mic_topics = get_mic_topics(scenario)
        mic_dfs = d.get_copied_datasets()
        flt_mic_dfs = topics_filter(mic_dfs, mic_topics)
        X_train, X_pos, X_neg, X_dfs = concat_datasets(flt_mic_dfs)

        # Scaling the data (important when features have different scales)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_pos_scaled = scaler.transform(X_pos)
        X_neg_scaled = scaler.transform(X_neg)

        # Initialize Isolation Forest
        iso_forest = IsolationForest()

        # Fit the model on the scaled training data
        iso_forest.fit(X_train_scaled)

        # Predict anomalies in the test set
        trn_pred = iso_forest.predict(X_train_scaled)

        # Predict anomalies in the test set
        pos_pred = iso_forest.predict(X_pos_scaled)

        # Predict anomalies in the test set
        neg_pred = iso_forest.predict(X_neg_scaled)

        # Count the number of detected anomalies
        # pos_anomalies = np.sum(pos_pred == -1)
        # print(f"Number of detected anomalies in positive: {pos_anomalies/len(pos_pred)}")
        #
        # # Count the number of detected anomalies
        # neg_anomalies = np.sum(neg_pred == -1)
        # print(f"Number of detected anomalies in negative: {neg_anomalies/len(neg_pred)}")

        all_preds = np.concatenate((trn_pred, pos_pred, neg_pred))
        dfs, p_dfs = convert_concat_to_datasets_n_prediction(X_dfs, all_preds, flt_mic_dfs)
        d.set_predictions(p_dfs[0], p_dfs[1], p_dfs[2])
        # ## print(predictions_information(d))
        trainings_preds, positives_preds, negatives_preds = d.get_predictions()
        # dfs[0][0].to_excel('example_df.xlsx', index=False)
        trainings_names, positives_names, negatives_names = d.get_names()
        multi_paths_preds = [zip(positives_names, positives_preds), zip(negatives_names, negatives_preds)]
        max_longest = find_max_longest_sequence(trainings_names, trainings_preds)
        mic_dict_preds = dict_sum_preds(multi_paths_preds, max_longest, dict_preds={})
        df['mic'] = sum_result(mic_dict_preds)



        # ---------------------    MAC    ---------------------------------


        mac_dfs = d.get_copied_datasets()
        d_mac_dfs = DS.drop_topics(mac_dfs, mic_topics, 'Active')
        # Scaling the data (important when features have different scales)
        n_d_mac_dfs = normalization(d_mac_dfs)
        mac_topics = most_influence_feature_by_pca(n_d_mac_dfs, n=27)
        flt_n_d_mac_dfs = topics_filter(n_d_mac_dfs, mac_topics)

        X_train, X_pos, X_neg, X_dfs = concat_datasets(flt_n_d_mac_dfs)
        # Initialize Isolation Forest
        iso_forest = IsolationForest()

        # Fit the model on the scaled training data
        iso_forest.fit(X_train_scaled)

        # Predict anomalies in the test set
        trn_pred = iso_forest.predict(X_train_scaled)

        # Predict anomalies in the test set
        pos_pred = iso_forest.predict(X_pos_scaled)

        # Predict anomalies in the test set
        neg_pred = iso_forest.predict(X_neg_scaled)

        # Count the number of detected anomalies
        # pos_anomalies = np.sum(pos_pred == -1)
        # print(f"Number of detected anomalies in positive: {pos_anomalies / len(pos_pred)}")
        #
        # # Count the number of detected anomalies
        # neg_anomalies = np.sum(neg_pred == -1)
        # print(f"Number of detected anomalies in negative: {neg_anomalies / len(neg_pred)}")

        all_preds = np.concatenate((trn_pred, pos_pred, neg_pred))
        dfs, p_dfs = convert_concat_to_datasets_n_prediction(X_dfs, all_preds, flt_mic_dfs)
        d.set_predictions(p_dfs[0], p_dfs[1], p_dfs[2])
        # ## print(predictions_information(d))
        trainings_preds, positives_preds, negatives_preds = d.get_predictions()
        trainings_names, positives_names, negatives_names = d.get_names()
        multi_paths_preds = [zip(positives_names, positives_preds), zip(negatives_names, negatives_preds)]
        max_longest = find_max_longest_sequence(trainings_names, trainings_preds)
        mac_dict_preds = dict_sum_preds(multi_paths_preds, max_longest, {})
        ## print("summarize Result for Macro features:")
        df['mac'] = sum_result(mac_dict_preds)

        for key in mic_dict_preds.keys():
            mic_dict_preds[key] += mac_dict_preds[key]

        ## print("\n\nUnion Result:")
        print("summarize Result for Union features:")
        filtered_mic_dict_preds = {k: v for k, v in mic_dict_preds.items() if k.startswith("norm") and v >= 1}
        # filtered_mic_dict_preds = {k: v for k, v in mic_dict_preds.items() if k.startswith("norm")}
        print(filtered_mic_dict_preds)

        df['union'] = sum_result(mic_dict_preds)
        group_names = extract_group_names(mic_dict_preds)
        df.index = group_names
        result = df
        # df.to_excel('C:\\Users\Avior\PycharmProjects\ROS-DBSCAN\\result.xlsx')
        if i == 0:
            s_result = result
            s2_result = result*result
        else:
            s_result += result
            s2_result += result*result

    avg = s_result / NFOLDS
    std = np.sqrt(s2_result / NFOLDS - avg * avg) * NFOLDS / (NFOLDS - 1)


    # Iterating over DataFrames and saving them to CSV
    save_with_title(avg, file_path, f"DataFrame avg {scenario}")
    save_with_title(std, file_path, f"DataFrame std {scenario}")

    print("average:")
    print(avg)
    print("std:")
    print(std)
    # avg = s_result / NFOLDS
    # std = np.sqrt(s2_result / NFOLDS - avg * avg) * NFOLDS / (NFOLDS - 1)
    # print("average:")
    # print(avg)
    # print("std:")
    # print(std)

        # # Apply PCA to reduce the data to 2D for visualization
        # pca = PCA(n_components=2)
        # X_test_pca = pca.fit_transform(X_test_scaled)

        # # Visualize the results (if your data is 2D)
        # plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap='coolwarm')
        # plt.title("Anomaly Detection using Isolation Forest (PCA-reduced data)")
        # plt.xlabel('PCA Component 1')
        # plt.ylabel('PCA Component 2')
        # plt.colorbar(label='Anomaly (-1) / Normal (1)')
        # plt.show()

        # # Apply PCA to reduce the data to 3D
        # pca = PCA(n_components=3)
        # X_test_pca_3d = pca.fit_transform(X_test_scaled)
        #
        # # 3D scatter plot
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(X_test_pca_3d[:, 0], X_test_pca_3d[:, 1], X_test_pca_3d[:, 2], c=y_pred, cmap='coolwarm')
        # ax.set_title("Anomaly Detection (PCA-reduced 3D data)")
        # ax.set_xlabel('PCA Component 1')
        # ax.set_ylabel('PCA Component 2')
        # ax.set_zlabel('PCA Component 3')
        # plt.show()

    #     # result = my_AEAD(d)





