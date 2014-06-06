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
        trainingErrorAverages
        testErrorAverages
        trainingErrorStdvs
        testErrorStdvs
        
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
      
        function [obj] = visualize(obj)
            %add something that gets variance/ error bars for all similar
            %data sizes
            obj.trainingErrorAverages = zeros(length(obj.numbersOfTargets),1);
            obj.testErrorAverages = zeros(length(obj.numbersOfTargets),1);
            obj.trainingErrorStdvs = zeros(length(obj.numbersOfTargets),1);
            obj.testErrorStdvs = zeros(length(obj.numbersOfTargets),1);
            for index = 1:length(obj.numbersOfTargets)
             
                indices = find(obj.TrainingDataSizes == obj.numbersOfTargets(index));
                obj.trainingErrorAverages(index) = mean(obj.TrainingErrors(indices));
                obj.trainingErrorStdvs(index) =  std(obj.TrainingErrors(indices));
                obj.testErrorAverages(index) = mean(obj.TestErrors(indices));
                obj.testErrorStdvs(index) = std(obj.TestErrors(indices));
            end
            %plot with bars avg and standard dev
            figure(1)
            errorbar(obj.numbersOfTargets,obj.trainingErrorAverages,obj.trainingErrorStdvs,'or');
            hold on;
            errorbar(obj.numbersOfTargets,obj.testErrorAverages,obj.testErrorStdvs,'og');
            xlabel('Training Data Size in number of Targets')
            yString = sprintf('Error ( %s )',obj.errorTypeName);
            ylabel(yString);
            legend('Training Error','Test Error')
            titleString = sprintf('Test and Training Errors for %s using parameter %s = %f',obj.methodName,obj.parameterName,obj.parameter);
            title(titleString);
            hold on;
%             figure(2)
%             plot(obj.TrainingDataSizes,obj.TrainingErrors,'or',obj.TrainingDataSizes,obj.TestErrors,'dg');
%             %hold on;
%             %plot(obj.TrainingDataSizes,obj.TrainingErrors,'-r',obj.TrainingDataSizes,obj.TestErrors,'-g');
%             xlabel('Training Data Size in number of Targets')
%             yString = sprintf('Error ( %s )',obj.errorTypeName);
%             ylabel(yString);
%             legend('Training Error','Test Error')
%             titleString = sprintf('Test and Training Errors for %s using parameter  %s = %f',obj.methodName,obj.parameterName,obj.parameter);
%             title(titleString);
%             
        end
            
%         function saveData(obj,directory,name,acc1, acc2, sizeoftraining)
%             file = fopen([directory name],'w');
%             fprintf(file,'Errors Data\n');
%             fprintf(file,'Size of Training Training Errors Testing Errors \n');
%             
%             fprintf(file,'%d \t \t \t%5.3f \t\t\t %5.3f \t',sizeoftraining,acc1, acc2');
%           
%             fclose(file);
%         end
%         
       
        
                
    end
            
            
        
end



