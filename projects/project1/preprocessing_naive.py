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

    # Select the features to be used given a first manual analysis
    new_dataset = np.delete(new_dataset, [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 18, 19,
                                          23, 24, 49, 51, 52, 53, 54, 55, 56, 57, 60, 64, 67, 68, 70, 71,
                                          74, 75, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 93, 96,
                                          98, 99, 101, 102, 105, 106, 111, 112, 113, 114, 115, 116, 117,
                                          118, 119, 120, 121, 122, 123, 124, 125, 126, 129, 130, 131, 132,
                                          133, 134, 135, 138, 139, 140, 141, 152, 153, 163, 164, 165, 166,
                                          167, 168, 169, 170, 171, 172, 174, 175, 176, 177, 178, 179, 180,
                                          181, 182, 183, 184, 185, 186, 187, 188, 189, 191, 194, 195, 196,
                                          197, 200, 201, 202, 203, 204, 205, 206, 208, 210, 211, 212, 214,
                                          215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
                                          228, 239, 245], axis=1)
    print(new_dataset.shape)

    means = np.nanmean(new_dataset, axis=0)

    # Find indices that you need to replace
    inds = np.where(np.isnan(new_dataset))

    # Place column means in the indices. Align the arrays using take
    new_dataset[inds] = np.take(means, inds[1])

    means = np.mean(new_dataset, axis=0)
    stds = np.std(new_dataset, axis=0)

    new_dataset = np.apply_along_axis(
        preprocessing_data_sample_naive, 1, new_dataset, means, stds)
    print(new_dataset.shape)
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
