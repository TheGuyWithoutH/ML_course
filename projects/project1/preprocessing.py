import numpy as np
from helpers import standardize


def default_nan(x): return 0 if x == np.nan else x


def replaceValueToInt(values, replacements):
    def replace(x):
        if x in values:
            return replacements[values.index(x)]
        else:
            return x
    return replace


def createCategory(categories):
    def category(x):
        if x in categories:
            arr = [0] * len(categories)
            arr[categories.index(x)] = 1
            return arr
        else:
            return [0] * len(categories)
    return category


def remplaceNaN(dataset, replacement, feature):
    new_dataset = np.copy(dataset)
    zeros = np.zeros(new_dataset[:, feature].shape)

    inds = np.where(np.isnan(new_dataset[:, feature]))

    new_dataset[inds][feature] = np.take(zeros, replacement)

    return new_dataset


rules = [
    default_nan,  # 10
    default_nan,  # 11
    default_nan,  # 14
    default_nan,  # 15
    default_nan,  # 16
    default_nan,  # 17
    default_nan,  # 20
    default_nan,  # 21
    default_nan,  # 22
    replaceValueToInt([77, 99, np.nan], [0, 0, 0]),  # 25
    replaceValueToInt([7, 9, np.nan], [7, 7, 7]),  # 26
    replaceValueToInt([88, 77, 99], [0, np.nan, np.nan]),  # 27
    replaceValueToInt([88, 77, 99], [0, np.nan, np.nan]),  # 28
    replaceValueToInt([88, 77, 99], [0, np.nan, np.nan]),  # 29
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 30
    replaceValueToInt([3, 7, 9], [0, 0, 0]),  # 31
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 32
    replaceValueToInt([7, 8, 9, np.nan], [np.nan, 5, np.nan, np.nan]),  # 33
    replaceValueToInt([2, 4, 3, 7, 9, np.nan], [1, 1, 0, 0, 0, 0]),  # 34
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 35
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 36
    replaceValueToInt([7, 9, np.nan], [np.nan, np.nan, np.nan]),  # 37
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 38
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 39
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 40
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 41
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 42
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 43
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 44
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 45
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 46
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 47
    replaceValueToInt([2, 4, 3, 7, 9, np.nan], [1, 1, 0, 0, 0, 0]),  # 48
    None,  # 50
    createCategory([1, 2, 3, 4, 5, 6, 7, 8]),  # 58
    replaceValueToInt([88, 99], [0, np.nan]),  # 59
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 61
    None,  # 62 ???
    None,  # 63 ???
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 65
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 66
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 69
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 72
    createCategory([1, 2, 3]),  # 73
    createCategory([1, 2, 3]),  # 76
    None,  # 77 ???
    replaceValueToInt([77, 99, np.nan], [np.nan, np.nan, 0]),  # 78
    replaceValueToInt([88, 77, 99, np.nan], [0, np.nan, np.nan, 0]),  # 79
    replaceValueToInt([2, 7, 9], [0, np.nan, np.nan]),  # 87
    None,  # 94 ???
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 95
    replaceValueToInt([1, 2, 3, 7, 9, np.nan], [
                      2, 1, 0, np.nan, np.nan, 0]),  # 97

    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 100
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 103
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 104
    replaceValueToInt([2, 7, 9], [0, 0, 0]),  # 107
    replaceValueToInt([2, 4, 3, 7, 9, np.nan], [1, 1, 0, 0, 0, 0]),  # 108
    replaceValueToInt([2, 4, 3, 7, 9, np.nan], [1, 1, 0, 0, 0, 0]),  # 109
    None,  # 110 ???
    createCategory([1, 2, 3, 4, 5, 6]),  # 127
    createCategory([1, 2, 3, 4, 5, 6]),  # 128
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 136
    replaceValueToInt([1, 2, 3, 4, 5, 7, 9, np.nan], [
                      4, 3, 2, 1, 0, np.nan, np.nan, 0]),  # 137
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 142
    None,  # 143 ???
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 144
    replaceValueToInt([97, 98, 99, np.nan], [
                      10, np.nan, np.nan, 0]),  # 145 ???
    replaceValueToInt([2, 7, np.nan], [0, 0, 0]),  # 146
    replaceValueToInt([88, 98, np.nan], [0, 0, 0]),  # 147
    replaceValueToInt([88, 98, np.nan], [0, 0, 0]),  # 148
    replaceValueToInt([88, 98, 99, np.nan], [0, 0, 0, 0]),  # 149
    replaceValueToInt([777, 888, 999, np.nan], [0, 0, 0, 0]),  # 150
    createCategory([1, 2, 3, 4, 5, 8]),  # 151 ???
    None,  # 154 ???
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 155
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 156
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 157
    replaceValueToInt([2, 3, 7, 9, np.nan], [1, 0, 0, 0, 0]),  # 158
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 159
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 160
    replaceValueToInt([2, 7, 9, np.nan], [0, 0, 0, 0]),  # 161
    createCategory([1, 2, 3, 4]),  # 162
    replaceValueToInt([7, 9, np.nan], [np.nan, np.nan, 0]),  # 173
    createCategory([1, 2, 3, 4]),  # 190
    createCategory([1, 2, 3, 4, 5]),  # 192
    createCategory([1, 2, 3, 4, 5]),  # 193
    createCategory([1, 2, 3, 4, 5]),  # 198
    replaceValueToInt([1, 2, 3, 4, 7, 9, np.nan], [
                      1, 1, 1, 0, 0, 0, 0]),  # 199

    replaceValueToInt([88, 77, 99], [0, np.nan, np.nan]),  # 207 ???
    replaceValueToInt([88, 77, 99], [0, np.nan, np.nan]),  # 209 ???
    replaceValueToInt([88, 77, 99, np.nan], [0, np.nan, np.nan, 0]),  # 213
    None,  # 229 c'est normal
    None,  # 230
    None,  # 231
    None,  # 232
    None,  # 233
    None,  # 234
    None,  # 235
    None,  # 236
    None,  # 237
    None,  # 238
    None,  # 240
    None,  # 241
    None,  # 242
    None,  # 243
    None,  # 244
    None,  # 246
    None,  # 247
    None,  # 248
    None,  # 249
    None,  # 250
    None,  # 251
    None,  # 252
    None,  # 253
    None,  # 254
    None,  # 255
    None,  # 256
    None,  # 257
    None,  # 258
    None,  # 259
    None,  # 260
    None,  # 261
    None,  # 262
    None,  # 263
    None,  # 264
    None,  # 265
    None,  # 266
    None,  # 267
    None,  # 268
    None,  # 269
    None,  # 270
    None,  # 271
    None,  # 272
    None,  # 273
    None,  # 274
    None,  # 275
    None,  # 276
    None,  # 277
    None,  # 278
    None,  # 279
    None,  # 280
    None,  # 281
    None,  # 282
    None,  # 283
    None,  # 284
    None,  # 285
    None,  # 286
    None,  # 287
    None,  # 288
    None,  # 289
    None,  # 290
    None,  # 291
    None,  # 292
    None,  # 293
    None,  # 294
    None,  # 295
    None,  # 296
    None,  # 297
    None,  # 298
    None,  # 299

    None,  # 300
    None,  # 301
    None,  # 302
    None,  # 303
    None,  # 304
    None,  # 305
    None,  # 306
    None,  # 307
    None,  # 308
    None,  # 309
    None,  # 310
    None,  # 311
    None,  # 312
    None,  # 313
    None,  # 314
    None,  # 315
    None,  # 316
    None,  # 317
    None,  # 318
    None,  # 319
    None,  # 320
]


def preprocessing_data_sample(x):
    # Copy the data to avoid modifying the original data
    transformed_input = np.array([])

    for feature in range(len(x)):
        if (rules[feature]):
            transformed_input = np.append(
                transformed_input, rules[feature](x[feature]))
        else:
            transformed_input = np.append(transformed_input, x[feature])
    return transformed_input


def preprocessing_dataset(dataset):
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

    new_dataset = np.apply_along_axis(
        preprocessing_data_sample, 1, new_dataset)

    # Replace all remaining NaN with mean
    means = np.nanmean(new_dataset, axis=0)
    # Find indices that you need to replace
    inds = np.where(np.isnan(new_dataset))
    # Place column means in the indices. Align the arrays using take
    new_dataset[inds] = np.take(means, inds[1])

    new_dataset, mean, std = standardize(new_dataset)
    return new_dataset, mean, std
