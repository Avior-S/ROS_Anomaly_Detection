import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_binomial_data(n, p, num_samples=1000):
    """ יוצר נתונים מתוך התפלגות בינומית """
    return np.random.binomial(n, p, num_samples)


def compute_confidence_interval(data, lower_percentile=5, upper_percentile=95):
    """ מחשב רווח סמך באמצעות אחוזונים """
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)
    median = np.median(data)
    return lower, median, upper


def plot_data_distribution(data, n, p):
    """ מציג את הנתונים באמצעות Box Plot ו-Bar Plot """
    plt.figure(figsize=(12, 5))

    # Box Plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data, color='skyblue')
    plt.title(f'Box Plot - Binomial({n}, {p})')
    plt.xlabel('Values')

    # Bar Plot (Histogram)
    plt.subplot(1, 2, 2)
    sns.histplot(data, bins=range(min(data), max(data) + 1), kde=False, color='salmon')
    plt.title(f'Bar Plot - Binomial({n}, {p})')
    plt.xlabel('Number of Successes')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()