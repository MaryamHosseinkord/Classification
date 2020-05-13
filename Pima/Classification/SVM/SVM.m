clc;
clear;
close all;

data = load('pima.dat');

inputData = data(:,1:end-1);
targetData = data(:,end);

indices = crossvalind('Kfold',targetData,10);

for i = 1:10
    t=tic;
    test = (indices == i); train = ~test;

    svmstruct=svmtrain(inputData(train),targetData(train));
    
    testOutput=svmclassify(svmstruct,inputData(test));
    
    timeVar(i) = toc(t);
    tableVar(i,:) = valueTable(testOutput,targetData(test));
    accVar(i) = ACC(tableVar(i,4) , tableVar(i,2) , tableVar(i,3) , tableVar(i,1));
    sensitivityVar(i) = sensitivity(tableVar(i,4) , tableVar(i,1));
    specificityVar(i) = specificity(tableVar(i,3) , tableVar(i,2));
    mccVar(i) = MCC(tableVar(i,4) , tableVar(i,2) , tableVar(i,3) , tableVar(i,1));
    f1Var(i) = F1_Score(tableVar(i,4) , tableVar(i,2) , tableVar(i,1));
end

mean_acc = mean(accVar)
std_acc = std(accVar)
mean_mcc = mean(mccVar)
mean_f1 = mean(f1Var)
mean_sensitivity = mean(sensitivityVar)
mean_specificity = mean(specificityVar)
mean_time = mean(timeVar)

figure;
x=0:0.01:1;
y=0:0.01:1;
plot(x,y,'-');
hold on;
plot(0,1,'or');
text(0,1, ['\leftarrow Perfect classification'])
hold on;
plot((1-mean_specificity),mean_sensitivity,'ro');    
axis([0 1 0 1]);
title('SVM-Pima ROC Plot');
xlabel('FPR');
ylabel('TPR');


