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
      
        function [obj] = prepareForVisualize(obj)
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
           

       
       
        
                
        end
            
            
        
    end
end



