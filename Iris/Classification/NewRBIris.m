clc;
clear;
close all;

data = load('Iris.dat');
inputData = data(:,1:end-1);
targetData = data(:,end);

indices = crossvalind('Kfold',targetData,10);

for i = 1:10
    t=tic;
    test = (indices == i); train = ~test;
    goal=0;
    spread=5;
    MaxNeuron=5;
    net = newrb(inputData(train,:)',targetData(train)',goal,spread,MaxNeuron);
    testOutput = sim(net,inputData(test,:)');
    
    
    [min, max]=MinMax(testOutput);
    testOutput= (testOutput-min(1,1))/(max(1,1)-min(1,1))*(3.49-(0.51))+ 0.51;
    testOutput = round(testOutput);

    tar=targetData(test);
   %Accuracy Calculation:
   error=0;
   for z=1:15
     if testOutput(z)~= tar(z)
    
    error=error+1;
   
   end
   end
    
    mseVar(i)=1-error/15;
    timeVar(i)=toc(t);
    %mseVar(i)=mse(targetData(test)-testOutput');
    %stdVar(i)=std(targetData(test)-testOutput');
end

mean_Acc = mean(mseVar)
[min, max]=MinMax(mseVar);
max_Acc = max
min_Acc = min
std_Acc = std(mseVar)
mean_time = mean(timeVar)