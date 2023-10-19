import numpy as np
from helpers import standardize


def preprocessing_data_sample_naive(x, means, stds):
    # Copy the data to avoid modifying the original data
    transformed_input = np.copy(x)
    temp_std = np.copy(stds)
    temp_std[temp_std == 0] = 1
    transformed_input = (transformed_input - means) / temp_std

    # Get top 50 variance features
    top50 = np.argsort(np.square(stds))[-200:]

    # Select the features to be used
    transformed_input = transformed_input[top50]

    return transformed_input


def preprocessing_dataset_naive(dataset):
    new_dataset = np.copy(dataset)
    means = np.nanmean(new_dataset, axis=0)

    # Find indices that you need to replace
    inds = np.where(np.isnan(new_dataset))

    # Place column means in the indices. Align the arrays using take
    new_dataset[inds] = np.take(means, inds[1])

    means = np.mean(new_dataset, axis=0)
    stds = np.std(new_dataset, axis=0)

    new_dataset = np.apply_along_axis(
        preprocessing_data_sample_naive, 1, new_dataset, means, stds)
    return new_dataset, means, stds


def preprocessing_dataset_test_naive(dataset, mean, std):
    new_dataset = np.copy(dataset)

    # Find indices that you need to replace
    inds = np.where(np.isnan(new_dataset))

    # Place column means in the indices. Align the arrays using take
    new_dataset[inds] = np.take(mean, inds[1])

    new_dataset = np.apply_along_axis(
        preprocessing_data_sample_naive, 1, new_dataset, mean, std)
    return new_dataset
