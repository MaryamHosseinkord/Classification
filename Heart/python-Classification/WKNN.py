import numpy as np
import math


# Independent Variables
data = np.genfromtxt('heart.dat')  # loading data set
train_test_rate = 0.7
k_factor = 5

# Data preparation / test set & train set
data = np.array(data)
(sample_size, temp) = np.shape(data)
feature_Num = temp - 1

threshold = math.floor(sample_size * train_test_rate)
train_data = data[:threshold, :]
test_data = data[threshold:, :]
(train_sample_size, _) = np.shape(train_data)
(test_sample_size, _) = np.shape(test_data)


# Defining a bias function
def biased_round(x):
    if x == 0.5:
        return 1
    else:
        return round(x)


# K_Nearest Neighbourhood Implementation
output_arr = []
for i in range(test_sample_size):
    temp_arr = (np.repeat(np.array([test_data[i, 0:feature_Num]]), train_sample_size, axis=0))
    dist_arr = np.sqrt(np.sum(pow((train_data[:, 0:feature_Num] - temp_arr), 2), 1))
    weights = 1 / np.sort(dist_arr)[0:k_factor]
    output = biased_round(np.sum(train_data[np.argsort(dist_arr)[0:k_factor], feature_Num] * weights) / np.sum(weights))
    output_arr.append(output)


# Evaluation of results:
FP = FN = TP = TN = 0
subtract_arr = test_data[:, feature_Num] - np.array(output_arr)

for i in range(test_sample_size):
    if subtract_arr[i] == -1:
        FP = FP + 1
    elif subtract_arr[i] == 1:
        FN = FN + 1
    elif subtract_arr[i] == 0 and test_data[i, feature_Num] == 1:
        TP = TP + 1
    else:
        TN = TN + 1

MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
F1 = (2 * TP) / ((2 * TP) + FP + FN)
ACC = (TP + TN) / (TP + FP + TN + FN)
Sens = TP / (TP + FN)
Spec = TN / (TN + FP)
classification_error = np.sum(np.absolute(subtract_arr)) / test_sample_size

print('end')
