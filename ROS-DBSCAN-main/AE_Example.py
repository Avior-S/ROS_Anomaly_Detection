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


# Create an imbalanced dataset
X, y = make_classification(n_samples=100000, n_features=32, n_informative=32,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.995, 0.005],
                           class_sep=0.5, random_state=0)


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# Check the number of records
print('The number of records in the training dataset is', X_train.shape[0])
print('The number of records in the test dataset is', X_test.shape[0])
print(f"The training dataset has {sorted(Counter(y_train).items())[0][1]} records for the majority class and {sorted(Counter(y_train).items())[1][1]} records for the minority class.")


# Keep only the normal data for the training dataset
X_train_normal = X_train[np.where(y_train == 0)]# Input layer
input = tf.keras.layers.Input(shape=(32,))# Encoder layers
encoder = tf.keras.Sequential([
  layers.Dense(16, activation='relu'),
  layers.Dense(8, activation='relu'),
  layers.Dense(4, activation='relu')])(input)# Decoder layers
decoder = tf.keras.Sequential([
      layers.Dense(8, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="sigmoid")])(encoder)# Create the autoencoder
autoencoder = tf.keras.Model(inputs=input, outputs=decoder)


# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mae')# Fit the autoencoder
history = autoencoder.fit(X_train_normal, X_train_normal,
          epochs=20,
          batch_size=64,
          validation_data=(X_test, X_test),
          shuffle=True)


# Predict anomalies/outliers in the training dataset
prediction = autoencoder.predict(X_test)# Get the mean absolute error between actual and reconstruction/prediction
prediction_loss = tf.keras.losses.mae(prediction, X_test)# Check the prediction loss threshold for 2% of outliers
loss_threshold = np.percentile(prediction_loss, 98)
print(f'The prediction loss threshold for 2% of outliers is {loss_threshold:.2f}')# Visualize the threshold
sns.histplot(prediction_loss, bins=30, alpha=0.8)
plt.axvline(x=loss_threshold, color='orange')
plt.show()


# Check the model performance at 2% threshold
threshold_prediction = [0 if i < loss_threshold else 1 for i in prediction_loss]# # Check the prediction performance
print(classification_report(y_test, threshold_prediction))