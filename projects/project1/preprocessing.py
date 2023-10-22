import numpy as np
from helpers import standardize


def default(x): return x
def default_nan(x): return 0 if x == np.nan else x

def replaceValueToInt(x, feature, value, replacement):
    for i in range(10):
        print(x[i,feature])
        print(x[i,feature]==value)
    return  np.where(x[:,feature]==value, replacement, x[:,feature])

def remplaceNaN(dataset, replacement, feature):
    new_dataset = np.copy(dataset)
    zeros = np.zeros(new_dataset[:,feature].shape)

   
    inds = np.where(np.isnan(new_dataset[:,feature]))

    new_dataset[inds][feature] = np.take(zeros, replacement)

    return new_dataset
rules = [
    default,  # Id
    default_nan,
    default_nan,
] + [lambda x: x] * 400


def preprocessing_data_sample(x):
    # Copy the data to avoid modifying the original data
    transformed_input = np.copy(x)

    # Select the features to be used
    np.delete(transformed_input, [1, 2, 3, 4, 5,
              6, 7, 8, 9, 12, 13, 18, 19, 23, 24, 38, 40])

    for feature in range(len(x)):
        if (rules[feature]):
            transformed_input[feature] == rules[feature](
                transformed_input[feature])
    return transformed_input


def preprocessing_dataset(dataset):
    new_dataset = np.copy(dataset)
    new_dataset = np.apply_along_axis(
        preprocessing_data_sample, 1, new_dataset)
    new_dataset, mean, std = standardize(new_dataset)
    return new_dataset, mean, std
