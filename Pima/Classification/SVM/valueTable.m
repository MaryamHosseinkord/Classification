function v = valueTable(testOutput,testTarget)

 n = numel(testOutput);
 FN = 0;
 FP = 0;
 TN = 0;
 TP = 0;

 for i=1:n
     if testOutput(i)==0 && testTarget(i)==0
        TN = TN + 1; 
     elseif testOutput(i)==1 && testTarget(i)==1
        TP = TP + 1;
     elseif testOutput(i) > testTarget(i)
        FP = FP + 1;    
     elseif testOutput(i) < testTarget(i)
        FN = FN + 1;
     end
 end
 
 v(1) = FN;
 v(2) = FP;
 v(3) = TN;
 v(4) = TP;
 
end

