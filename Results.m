%Evan Racah
%4/21/14
%Results class for calculating accuracies
%when given the results
classdef Results
    properties
        TrainingAccuracies
        TestAccuracies
        TrainingDataSizes
        LabelledAnswers
       
        count;
    end
    methods
        function obj=Results(labels) %constructor
            obj.LabelledAnswers = labels;
            obj.count = 0;
            
        end
        function addNewResults(obj,size, trainingResults, trainingPermutation, testingResults, testingPermutation )
            obj.count = obj.count + 1;
            obj.TrainingDataSizes(obj.count) = size; %appending vector
            obj.TrainingAccuracies(obj.count) = obj.getAccuracy(trainingResults, trainingPermutation);
            obj.TestAccuracies(obj.count) = obj.getAccuracy(testingResults, testingPermutation);
            disp(obj.TrainingAccuracies)
        end
        function acc=getAccuracy(obj, results, permutation)
           %[r, c] = size(results);
           %use r,1 if results is column vector and 1,c if row vector
           acc = mean(obj.calcPercentError(results,obj.LabelledAnswers(permutation))); %for now. Not sure how to calculate accuracy for regression
        end
        function pErrors=calcPercentError(obj,results,realThing)
            pErrors = (results - realThing)./realThing;  %dot does elementwise division
        end
        
                
    end
            
            
        
end



%todo write test driver for this class