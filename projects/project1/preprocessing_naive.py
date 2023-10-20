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
    # transformed_input = transformed_input[[14,  26, 157, 248, 246,  58, 249,  34, 231, 247, 230, 232, 238, 48, 144,  66,  69,  45,
    #                                       147, 162, 154,  65,  59,  38, 173, 190, 234, 153,  61, 289, 150, 103,  39,  27, 133, 138,
    #                                        223, 256, 213,  97,  44, 158, 315,  57, 178,  77, 109, 127,  52, 188, 137,  35, 216, 257,
    #                                        72,  37, 142,  50, 176, 17,  71, 140,  94, 313, 209,  20, 128, 254,  95,  15, 304,  70,
    #                                        43, 108, 170, 197,  29,  42, 193, 100, 175, 253, 252, 207, 145, 233, 302,  33, 258,  79,
    #                                        47, 159,  36, 110,   7,   8, 259, 194,  84, 229]]
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
