function y = ROC( TP , FN , FP , TN )

TPR = TP / ( TP + FN );

FPR = FP / ( TN + FP );

y(1) = TPR ;
y(2) = FPR ;

end

