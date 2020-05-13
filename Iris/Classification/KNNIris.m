% K-Nearest Neighbor
% Done by Maryam Hosseinkord
% Date: 20/12/2013


clear
clc

disp('DataSet: Iris');
disp('Numbers displayed below, are the average results of 10 runs');

KK=[1 3 5]; % K-NN Parameter
load Iris.dat
[m,n] = size(Iris);

Vect = randperm(150);%load('test vector.txt');
DataSet = zeros(m,n);
for i=1:m
    DataSet(i,:)=Iris(Vect(i),:);
    
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


Part= m/10;

for i=1:10
    tic
    TrainSet = DataSet;
    TestSet = DataSet(Part*(i-1)+1 :Part*i, :);
    TestInput = TestSet(:,1:4);
    TestTarget = TestSet(:,5);
    
    TrainSet(Part*(i-1)+1 :Part*i, :)=[];
    TrainInput = TrainSet(:,1:4);
    TrainTarget = TrainSet(:,5);
    
    for j= 1:15
        
          for z= 1: 135
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
          Class=[0 0 0];
    
    
         for l=1: K
              switch DecisionGroup(l)
                   case 1
                      Class(1)=Class(1) + 1;
                   case 2
                      Class(2)=Class(2) + 1;
                   case 3
                      Class(3)=Class(3) + 1;
              end
         end
    
         Max=max(Class);
         Ind=find(Class(:,:)==Max);
         OutPut(j)=Ind(1,1);
        
    end
   
    error=0;
    
   %Accuracy Calculation:
   for z=1:15
   if OutPut(z)~= TestTarget(z)
    
    error=error+1;
   
   end
   end
    
   Accuracy(i)=1-error/15;
   ti(i)=toc;
    
    
end




Time= mean(ti);
Mean= mean(Accuracy);
STD=std(Accuracy);
Best=max(Accuracy);
Worst=min(Accuracy);


Result=[Mean STD Best Worst Time];
disp('     Mean       Std      Best      Worst     Time');
disp(Result);

end