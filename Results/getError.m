function error=getError(guess, answer,errorType)
           error = norm(answer - guess, 2); %for now. Not sure how to calculate accuracy for regression
           if (strcmp(errorType,'MSE')) == 0
               error = error^2 / length(guess);
          
%            elseif (strcmp(errorType,'L2Norm')) == 0
        
          
           end
        end