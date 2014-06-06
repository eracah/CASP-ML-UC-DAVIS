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
        errorTypeName;
        numbersOfTargets
        
    end
    methods
        function obj=Results(methodName, parameterName, errorType,numTargetsArray) %constructor
            obj.numbersOfTargets = numTargetsArray;
            obj.methodName = methodName;
            obj.parameterName = parameterName;
            obj.errorTypeName = errorType;
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
            %add something that gets variance/ error bars for all similar
            %data sizes
            trainingErrorAverages = zeros(length(obj.numbersOfTargets),1);
            testErrorAverages = zeros(length(obj.numbersOfTargets),1);
            trainingErrorStdvs = zeros(length(obj.numbersOfTargets),1);
            testErrorStdvs = zeros(length(obj.numbersOfTargets),1);
            for index = 1:length(obj.numbersOfTargets)
             
                indices = find(obj.TrainingDataSizes == obj.numbersOfTargets(index));
                trainingErrorAverages(index) = mean(obj.TrainingErrors(indices));
                trainingErrorStdvs(index) =  std(obj.TrainingErrors(indices));
                testErrorAverages(index) = mean(obj.TestErrors(indices));
                testErrorStdvs(index) = std(obj.TestErrors(indices));
            end
            %plot with bars avg and standard dev
            figure(1)
            errorbar(obj.numbersOfTargets,trainingErrorAverages,trainingErrorStdvs,'or');
            hold on;
            errorbar(obj.numbersOfTargets,testErrorAverages,testErrorStdvs,'og');
            figure(2)
            plot(obj.TrainingDataSizes,obj.TrainingErrors,'or',obj.TrainingDataSizes,obj.TestErrors,'dg');
            %hold on;
            %plot(obj.TrainingDataSizes,obj.TrainingErrors,'-r',obj.TrainingDataSizes,obj.TestErrors,'-g');
            xlabel('Training Data Size in number of Targets')
            yString = sprintf('Error ( %s )',obj.errorTypeName);
            ylabel(yString);
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



