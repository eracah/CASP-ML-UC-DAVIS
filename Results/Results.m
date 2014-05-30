%Evan Racah
%4/21/14
%Results class for calculating Errors
%when given the results
classdef Results
    properties
        TrainingErrors;
        TestErrors;
        TestGuesses
        TrainingDataSizes;
        TrainingGuesses
        count;
        methodName;
        parameter;
        parameterName;
        
    end
    methods
        function obj=Results(methodName, parameterName) %constructor
            obj.methodName = methodName;
            obj.parameterName = parameterName;
            obj.count = 0;
            
        end
        %return obj for all functions that chanbge data members of obj
        function [obj] = addNewResults(obj,trainingSize, trainingError, trainingGuesses, testError, testGuesses,parameterUsed)
            obj.count = obj.count + 1;
            obj.TrainingDataSizes(obj.count) = trainingSize;
            obj.TrainingErrors(obj.count) = trainingError;
            obj.TestErrors(obj.count) = testError;
            obj.TrainingGuesses{obj.count} = trainingGuesses;
            obj.TestGuesses{obj.count} = testGuesses;
            obj.parameter = parameterUsed;
            
            %obj.saveData('./Results/','results.txt',obj.TrainingErrors(obj.count),obj.TestErrors(obj.count),length(trainingPermutation));
        end
      
        function visualize(obj)
            plot(obj.TrainingDataSizes,obj.TrainingErrors,'-r',obj.TrainingDataSizes,obj.TestErrors,'-g');
            xlabel('Training Data Size in number of Targets')
            ylabel('Error')
            legend('Training Error','Test Error')
            titleString = sprintf('Test and Training Errors for %s using parameter %s = %f',obj.methodName,obj.parameterName,obj.parameter);
            title(titleString);
            
        end
            
        function saveData(obj,directory,name,acc1, acc2, sizeoftraining)
            file = fopen([directory name],'w');
            fprintf(file,'Errors Data\n');
            fprintf(file,'Size of Training Training Errors Testing Errors \n');
            
            fprintf(file,'%d \t \t \t%5.3f \t\t\t %5.3f \t',sizeoftraining,acc1, acc2');
          
            fclose(file);
        end
        
       
        
                
    end
            
            
        
end



