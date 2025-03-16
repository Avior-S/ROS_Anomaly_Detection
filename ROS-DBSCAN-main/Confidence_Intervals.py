import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def compute_confidence_interval(data, lower_percentile=16.5, upper_percentile=83.5):
    """ מחשב רווח סמך באמצעות אחוזונים """
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)
    median = np.median(data)
    return lower, median, upper

def plot_confidence_intervals(intervals, scenario_labels, scenario):
    """ Displays confidence intervals using a Bar Plot """
    plt.figure(figsize=(8, 5))
    lower_bounds, medians, upper_bounds = zip(*intervals)
    plt.errorbar(scenario_labels, medians,
                 yerr=[np.array(medians) - np.array(lower_bounds), np.array(upper_bounds) - np.array(medians)],
                 fmt='o', capsize=5, color='blue', label='67% Bayesian CI')
    plt.title(f'{scenario} Bayesian Confidence Intervals for Anomaly Detection Rate')
    plt.xlabel('Scenarios')
    plt.ylabel('Detection Rate')
    plt.legend()
    plt.savefig("plots\\"+scenario+".png")
    print(f"Plot saved to plots\\{scenario}.png")
    # plt.show()


def plot_all_confidence_intervals(intervals, scenario_labels, all_results, scenario=None, save_path=None):
    """ Displays confidence intervals using a Box Plot along with the results of 10-fold cross-validation """
    plt.figure(figsize=(12, 8))

    # Plot the box plot for the 10-fold results
    plt.boxplot(all_results, positions=range(len(scenario_labels)), widths=0.5, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                whiskerprops=dict(color='blue'),
                flierprops=dict(marker='o', color='red', markersize=5, label='Outliers'))

    # # Plot individual points for each sample (10-fold results)
    # for i, scenario_results in enumerate(all_results):
    #     # Scatter points for each sample (fold)
    #     plt.scatter([i] * len(scenario_results), scenario_results, color='black', alpha=0.6,
    #                 label='Sample Points' if i == 0 else "")

    # Extract lower bounds, medians, and upper bounds for the confidence intervals
    lower_bounds, medians, upper_bounds = zip(*intervals)

    # Plot the confidence intervals as error bars
    plt.errorbar(range(len(scenario_labels)), medians,
                 yerr=[np.array(medians) - np.array(lower_bounds), np.array(upper_bounds) - np.array(medians)],
                 fmt='o', capsize=5, color='blue', label='67% Bayesian CI')

    # Add labels and title
    plt.title(f'{scenario} Bayesian Confidence Intervals and 10-Fold Results for Anomaly Detection Rate')
    plt.xlabel('Scenarios')
    plt.ylabel('Detection Rate')
    plt.xticks(range(len(scenario_labels)), scenario_labels)  # Label for each scenario
    plt.legend()

    # Save the plot if a save path is provided
    plt.savefig("plots\\all_points_"+scenario+".png")
    print(f"Plot saved to plots\\{scenario}.png")

    # Show the plot
    plt.show()


# def plot_all_confidence_intervals(intervals, scenario_labels, lists_all_results, scenario):
#     """ Displays confidence intervals using a Box Plot along with the results of 10-fold cross-validation """
#     plt.figure(figsize=(10, 6))
#
#     # Plot the box plot for the 10-fold results
#     plt.boxplot(lists_all_results, positions=range(len(scenario_labels)), widths=0.5, patch_artist=True,
#                 boxprops=dict(facecolor='lightblue', color='blue'),
#                 whiskerprops=dict(color='blue'), flierprops=dict(marker='o', color='red', markersize=5))
#
#     # Extract lower bounds, medians, and upper bounds for the confidence intervals
#     lower_bounds, medians, upper_bounds = zip(*intervals)
#
#     # Plot the confidence intervals as error bars
#     plt.errorbar(range(len(scenario_labels)), medians,
#                  yerr=[np.array(medians) - np.array(lower_bounds), np.array(upper_bounds) - np.array(medians)],
#                  fmt='o', capsize=5, color='blue', label='67% Bayesian CI')
#
#     # Add labels and title
#     plt.title(f'{scenario} Bayesian Confidence Intervals and 10-Fold Results for Anomaly Detection Rate')
#     plt.xlabel('Scenarios')
#     plt.ylabel('Detection Rate')
#     plt.xticks(range(len(scenario_labels)), scenario_labels)  # Label for each scenario
#     plt.legend()
#
#     # Save the plot if a save path is provided
#
#     plt.savefig("plots\\all_points"+scenario+".png")
#     print(f"Plot saved to plots\\{scenario}.png")

    # Show the plot
    # plt.show()

