# Synthetic dataset
from sklearn.datasets import make_classification# Data processing
import pandas as pd
import numpy as np
from collections import Counter# Visualization
import matplotlib.pyplot as plt
import seaborn as sns# Model and performance
import tensorflow as tf
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import ROS_DBSCAN as rd


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


def AEAD(datasets, feat=27):
    trainings, positives, negatives = datasets
    X_train, X_test, Xdfs = rd.concat_datasets(datasets)
    # print('The number of records in the training dataset is', X_train.shape[0])
    # print('The number of records in the test dataset is', X_test.shape[0])

    # Keep only the normal data for the training dataset
    if feat > 16:
        input = tf.keras.layers.Input(shape=(27,))# Encoder layers
        encoder = tf.keras.Sequential([
          layers.Dense(16, activation='relu'),
          layers.Dense(8, activation='relu'),
          layers.Dense(4, activation='relu')])(input)# Decoder layers
        decoder = tf.keras.Sequential([
              layers.Dense(8, activation="relu"),
              layers.Dense(16, activation="relu"),
              layers.Dense(27, activation="sigmoid")])(encoder)# Create the autoencoder
        autoencoder = tf.keras.Model(inputs=input, outputs=decoder)
    else:
        input = tf.keras.layers.Input(shape=(3,))# Encoder layers
        encoder = tf.keras.Sequential([
          layers.Dense(2, activation='relu')])(input)# Decoder layers
        decoder = tf.keras.Sequential([
              layers.Dense(3, activation="sigmoid")])(encoder)# Create the autoencoder
        autoencoder = tf.keras.Model(inputs=input, outputs=decoder)


    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mae')# Fit the autoencoder
    history = autoencoder.fit(X_train[:int(len(X_train)*0.8)], X_train[:int(len(X_train)*0.8)],
              epochs=20,
              batch_size=64,
              validation_data=(X_train[int(len(X_train)*0.8):], X_train[int(len(X_train)*0.8):]),
              shuffle=True)

    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # Predict anomalies/outliers in the training dataset
    train_prediction = autoencoder.predict(X_train)# Get the mean absolute error between actual and reconstruction/prediction
    train_prediction_loss = tf.keras.losses.mae(train_prediction, X_train)# Check the prediction loss threshold for 2% of outliers
    loss_threshold = np.percentile(train_prediction_loss, 95)
    # print(f'The prediction loss threshold for 5% of outliers is {loss_threshold:.2f}')# Visualize the threshold
    sns.histplot(train_prediction_loss, bins=30, alpha=0.8)
    # plt.axvline(x=loss_threshold, color='orange')
    # plt.show()

    test_prediction = autoencoder.predict(
        X_test)  # Get the mean absolute error between actual and reconstruction/prediction
    test_prediction_loss = tf.keras.losses.mae(test_prediction, X_test)


    # Check the model performance at 2% threshold
    predictions = [1 if i < loss_threshold else 0 for i in train_prediction_loss] # Check the prediction performance
    predictions.extend([1 if i < loss_threshold else 0 for i in test_prediction_loss])
    dfs, n_dfs = convert_concat_to_datasets_n_prediction(Xdfs, predictions, datasets)
    return dfs, n_dfs