function y = MCC( TP , FP , TN , FN )

y = ( ( TP * TN ) - ( FP * FN ) ) / sqrt( ( TP + FP ) * ( TP + FN ) * ( TN + FP ) * ( TN + FN ) );

end

