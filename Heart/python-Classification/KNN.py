import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance

# Independent Variables
data = np.genfromtxt('heart.dat')  # loading data set
train_test_rate = 0.7
k_factor = 3
# -----------------------------


# Data preparation / test set & train set
data = np.array(data)
(sample_size, temp) = np.shape(data)
feature_Num = temp - 1

threshold = math.floor(sample_size * train_test_rate)
train_data = data[:threshold, :]
test_data = data[threshold:, :]
(train_sample_size, _) = np.shape(train_data)
(test_sample_size, _) = np.shape(test_data)

#print(np.random.permutation(sample_size))

# K Nearest Neighbourhood Implementation
output_arr = []
for i in range(test_sample_size):
    temp_arr = (np.repeat(np.array([test_data[i, 0:feature_Num]]), train_sample_size, axis=0))
    dist_arr = np.sqrt(np.sum(pow((train_data[:, 0:feature_Num] - temp_arr), 2), 1))
    output = round(np.average(train_data[np.argsort(dist_arr)[0:k_factor], feature_Num]))
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
classification_error = np.sum(np.absolute( subtract_arr )) / test_sample_size



print('end')

"""

MCC(i) = ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
F1(i) = (2 * TP) / ((2 * TP) + FP + FN);
ACC(i) = (TP + TN) / (TP + FP + TN + FN);
Sens(i) = TP / (TP + FN);
Spec(i) = TN / (TN + FP);



Vect = random.randint(sample_Size); # load('test vector.txt');

DataSet = zeros(sample_Size, feat_Num);
for i=1:sample_Size
DataSet(i,:)=Heart(Vect(i),:);

end

for nearest=1:3

K = KK(nearest);

if K == 1
    disp('Classifier: 1-NN');
elseif
K == 3
disp('Classifier: 3-NN');
else
disp('Classifier: 5-NN');
end

% 10 fold cross validation
Part = m / 10;

for i=1:10 % run
tic

FP = 0;
FN = 0;
TP = 0;
TN = 0;

TrainSet = DataSet;
TestSet = DataSet(Part * (i - 1) + 1:Part * i,:);
TestInput = TestSet(:, 1: 13);
TestTarget = TestSet(:, 14);
TrainSet(Part * (i - 1) + 1: Part * i,:)=[];
TrainInput = TrainSet(:, 1: 13);
TrainTarget = TrainSet(:, 14);

for j= 1:27

for z= 1: 243
diff = TestInput(j,:)- TrainInput(z,:);
Distance(z) = sqrt(sum(diff. ^ 2));
end

Dist = Distance;
for k=1: K
a = min(Dist);
temp = find(Distance(:,:) == min(Dist));
Index(k) = temp(1, 1);
b = find(Dist(:,:) == min(Dist));
Dist(:, b)=[];
end

DecisionGroup = TrainTarget(Index);
Class = [0 0];
for l=1: K
switch
DecisionGroup(l)
case
0
Class(1) = Class(1) + 1;
case
1
Class(2) = Class(2) + 1;
end
end

Max = max(Class);
Ind = find(Class(:,:) == Max);
OutPut(j) = Ind(1, 1) - 1;

end

% Hypothesis
Evaluation:
AccVect = TestTarget - OutPut
';

for count= 1:27
if AccVect(count) == -1
    FP = FP + 1;
elseif
AccVect(count) == 1
FN = FN + 1;
elseif
AccVect(count) == 0
if TestTarget(count) == 1
    TP = TP + 1;
else
    TN = TN + 1;
end
end
end

ti(i) = toc;
MCC(i) = ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
F1(i) = (2 * TP) / ((2 * TP) + FP + FN);
ACC(i) = (TP + TN) / (TP + FP + TN + FN);
Sens(i) = TP / (TP + FN);
Spec(i) = TN / (TN + FP);

end % end
of
run

Mean = mean(ACC);
STD = std(ACC);
Time = mean(ti);
MCC = mean(MCC);
F1 = mean(F1);
Sensitivity = mean(Sens);
Specificity = mean(Spec);

Result = [Mean STD MCC F1 Sensitivity Specificity Time];
disp('     Mean       Std       MCC       F1       Sens      Spec      Time');
disp(Result)

TPR(nearest) = Sensitivity;
FPR(nearest) = 1 - Specificity;

end

jj = [1, 3, 5];
figure;
x = 0:0.01: 1;
y = 0:0.01: 1;
plot(x, y, '-');
hold
on;
plot(0, 1, 'or');
text(0, 1, ['\leftarrow Perfect classification'])
hold
on;
for i=1:3
plot(FPR(i), TPR(i), 'xr');

for i=1:3
text(FPR(i), TPR(i), [num2str(jj(i))], 'FontSize', 9);

end
hold
on;
end

axis([0 1 0 1]);
title('ROC Plot');
xlabel('FPR');
ylabel('TPR');
"""
