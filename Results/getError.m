function error=getError(guess, answer,errorType)
           fprintf('mean guess %f \n',mean(guess));
           fprintf('mean answer %f \n',mean(answer));
           
           error = norm(answer - guess, 2); %for now. Not sure how to calculate accuracy for regression
           error = error^2 / length(guess);
           fprintf ('error %f \n',error);
        
          
end
        