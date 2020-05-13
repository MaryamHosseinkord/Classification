% K-Nearest Neighbor
% Done by Maryam Hosseinkord
% Date: 20/12/2013

clear
clc

disp('DataSet: Statlog Heart');
disp('Numbers displayed below, are the average results of 10 runs:');
disp('   ');

KK=[1 3 5]; % K-NN Parameter
load Heart.dat
[m,n] = size(Heart);


% Dataset re-arrangement
Vect =  randperm(270);%load('test vector.txt');
DataSet = zeros(m,n);
for i=1:m
    DataSet(i,:)=Heart(Vect(i),:);
    
end


for nearest=1:3
    
K=KK(nearest);

if K==1
    disp('Classifier: 1-NN');
elseif K==3
     disp('Classifier: 3-NN');
else
     disp('Classifier: 5-NN');
end



% 10 fold cross validation
Part= m/10;

for i=1:10 %run
    tic

    FP=0;
    FN=0;
    TP=0;
    TN=0;
        
    TrainSet = DataSet;
    TestSet = DataSet(Part*(i-1)+1 :Part*i, :);
    TestInput = TestSet(:,1:13);
    TestTarget = TestSet(:,14);
    TrainSet(Part*(i-1)+1 :Part*i, :)=[];
    TrainInput = TrainSet(:,1:13);
    TrainTarget = TrainSet(:,14);
    
    for j= 1:27 
        
          for z= 1: 243
              diff= TestInput(j,:)- TrainInput(z,:);
              Distance(z)=sqrt(sum(diff .^2));
          end
        
          Dist=Distance;
          for k=1: K
              a=min(Dist);
              temp=find(Distance(:,:)== min(Dist));
              Index(k)=temp(1,1);
              b = find(Dist(:,:)== min(Dist));
              Dist(:,b)=[];
          end
    
          DecisionGroup = TrainTarget(Index);
          Class=[0 0];
          for l=1: K
              switch DecisionGroup(l)
                   case 0
                      Class(1)=Class(1) + 1;
                   case 1
                      Class(2)=Class(2) + 1;
              end
          end
    
          Max=max(Class);
          Ind=find(Class(:,:)==Max);
          OutPut(j)=Ind(1,1)-1;
        
    end 
        
     
   % Hypothesis Evaluation:
   AccVect=TestTarget-OutPut';
     
   for count= 1:27
       if AccVect(count)==-1
           FP=FP+1;
       elseif AccVect(count)==1
           FN=FN+1;
       elseif AccVect(count)==0
             if TestTarget(count)==1
                 TP=TP+1;
             else
                 TN=TN+1;
             end
       end
   end
   
   ti(i)=toc; 
   MCC(i)=((TP*TN)-(FP*FN))/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
   F1(i)= (2*TP)/((2*TP)+FP+FN);
   ACC(i)= (TP+TN)/(TP+FP+TN+FN);
   Sens(i)= TP/(TP+FN);
   Spec(i)= TN/(TN+FP);
   
end % end of run

Mean= mean(ACC);
STD=std(ACC);
Time= mean(ti);
MCC=mean(MCC);
F1=mean(F1);
Sensitivity=mean(Sens);
Specificity=mean(Spec);


Result=[Mean STD MCC F1 Sensitivity Specificity Time];
disp('     Mean       Std       MCC       F1       Sens      Spec      Time');
disp(Result)


TPR(nearest)=Sensitivity;
FPR(nearest)=1-Specificity;

end

jj=[1,3,5];
figure;
x=0:0.01:1;
y=0:0.01:1;
plot(x,y,'-');
hold on;
plot(0,1,'or');
text(0,1, ['\leftarrow Perfect classification'])
hold on;
for i=1:3
    plot(FPR(i),TPR(i),'xr');  
    
    for i=1:3
        text(FPR(i), TPR(i), [ num2str(jj(i))  ],'FontSize', 9);
        
    end
    hold on;
end

axis([0 1 0 1]);
title('ROC Plot');
xlabel('FPR');
ylabel('TPR');