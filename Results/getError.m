function error=getError(guess, answer)
           error =norm(answer - guess, 2); %for now. Not sure how to calculate accuracy for regression
        end