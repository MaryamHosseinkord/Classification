clc;
clear;
close all;

data = load('Iris.dat');

inputData = data(:,1:end-1);
targetData = data(:,end);
fSpace = size(inputData,2);

indices = crossvalind('Kfold',targetData,10);

for i = 1:10
    t=tic;
    test = (indices == i); train = ~test;
    
    dataTrain = inputData(train,:);
    trainSize = size(dataTrain,1);
    dataTest = inputData(test,:);
    testSize = size(dataTest,1);
    centers = randi(trainSize,[1,fSpace]);
  
    for j=1:fSpace
        for k=1:fSpace
            dictanceCenter(j,k) = norm(dataTrain(centers(j),:)-dataTrain(centers(k),:));
        end
    end
    dictanceCenter = triu(dictanceCenter);
    
    mDistance = max(max(dictanceCenter));
    
    sigma = mDistance / sqrt(fSpace);
    
    phi = zeros(trainSize,fSpace);
    for j=1:trainSize
        for k=1:fSpace
            phi(j,k) = exp(((-fSpace/mDistance.^2)*(norm(dataTrain(j,:)-dataTrain(centers(k),:)).^2)));
        end
    end
        
    w = pinv(phi)*targetData(train);
    
    phiTest = zeros(testSize,fSpace);
    for j=1:testSize
        for k=1:fSpace
            phiTest(j,k) = exp(((-fSpace/mDistance.^2)*(norm(dataTest(j,:)-dataTrain(centers(k),:)).^2)));
        end
    end
    
    yTest = zeros(1,testSize);
    for j=1:testSize
        for k=1:fSpace
            yTest(j) = sum(w(k)*phiTest(j,k));
        end
    end

    yTestNormal = zeros(1,testSize);
    for j=1:testSize
        yTestNormal(j) =round(((yTest(j) - min(yTest))/(max(yTest) - min(yTest)))*(3.49-0.51)+0.51);
    end
    
    testInput = targetData(test);
    faultNo= 0;
    for j=1:testSize
       if testInput(j) ~= yTestNormal(j)           
           faultNo = faultNo + 1;
       end
    end
    
    timeVar(i)=toc(t);
    accVar(i)=1-(faultNo/testSize);   
end

mean_acc = mean(accVar)
max_acc = max(accVar)
min_acc = min(accVar)
std_acc = std(accVar)
mean_time = mean(timeVar)