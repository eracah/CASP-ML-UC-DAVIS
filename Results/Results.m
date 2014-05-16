%Evan Racah
%4/21/14
%Results class for calculating accuracies
%when given the results
classdef Results
    properties
        TrainingAccuracies;
        TestAccuracies;
        TrainingDataSizes;
        LabelledAnswers;
        count;
    end
    methods
        function obj=Results(labels) %constructor
            obj.LabelledAnswers = labels;
            obj.count = 0;
            numPoints = 200;
            obj.TrainingDataSizes = zeros(numPoints,1);
            obj.TrainingAccuracies = zeros(numPoints,1);
            obj.TestAccuracies =zeros(numPoints,1);
        end
        function addNewResults(obj,size, trainingResults, trainingPermutation, testingResults, testingPermutation )
            obj.count = obj.count + 1;
            obj.TrainingDataSizes(obj.count) = size;
            obj.TrainingAccuracies(obj.count) = obj.getAccuracy(trainingResults, trainingPermutation);
            obj.TestAccuracies(obj.count) = obj.getAccuracy(testingResults, testingPermutation);
            obj.saveData('./Results/','results.txt',obj.TrainingAccuracies(obj.count),obj.TestAccuracies(obj.count),length(trainingPermutation));
        end
        function acc=getAccuracy(obj, results, permutation)
           pErrors = (results - obj.LabelledAnswers(permutation))./obj.LabelledAnswers(permutation);
           acc = mean(pErrors); %for now. Not sure how to calculate accuracy for regression
        end
      
        function saveData(obj,directory,name,acc1, acc2, sizeoftraining)
            file = fopen([directory name],'w');
            fprintf(file,'Accuracies Data\n');
            fprintf(file,'Size of Training Training Accuracies Testing Accuracies \n');
            
            fprintf(file,'%d \t \t \t%5.3f \t\t\t %5.3f \t',sizeoftraining,acc1, acc2');
          
            fclose(file);
        end
        
                
    end
            
            
        
end



%todo write test driver for this class