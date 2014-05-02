%Evan Racah
%5/1/14
%Data class for extracting the data
classdef Data
    properties
        outputIndx
        startFeatureIndx
        endFeatureIndx
        targets %cell array of various targets
        rawData %expermentata object
        featureNames  %1x42 cell array of string names of features
        dataFeatureValues %matrix of nmodels x nfeatures with all data
        dataLabels %nmodels x 1 with label for each target
        numFeatures
        totModels
        
    end
    methods
        function obj=Data(DotMatObject, outputName, startFeature, endFeature)
            obj.rawData = DotMatObject; %ExperimentData object
            obj.targets = obj.rawData.targetsData;
            testFields = obj.targets{1}.fields;
            indices = find(ismember(testFields,{outputName;startFeature; endFeature})); %finds index for various features
            obj.outputIndx = indices(1);
            obj.startFeatureIndx = indices(2);
            obj.endFeatureIndx = indices(3);
            obj.dataFeatureValues = [];
            obj.numFeatures = obj.endFeatureIndx - obj.startFeatureIndx + 1
            
        end
        function [totModels, dataFeatureValues, dataLabels]=extractData(obj,numTargets,numModelsPerTarget) %each target has varying number of models but for now we will take the first k models of each target where k = numModelsPerTarget
            totModels = numTargets*numModelsPerTarget;
            dataFeatureValues = zeros(obj.totModels,obj.numFeatures);
            dataLabels = zeros(obj.totModels,1);
            i = 1;
            while (i < totModels) %loops thru each target and grabs data for first numModelsPerTarget
                k = (i:numModelsPerTarget + i - 1)';
                dataFeatureValues(k,:) = obj.targets{i+110}.values(1:numModelsPerTarget,obj.startFeatureIndx:obj.endFeatureIndx);
                dataLabels(k) = obj.targets{i + 110}.values(1:numModelsPerTarget,obj.outputIndx);
                i = i + numModelsPerTarget;
            end
           
        end
        function [trainingData,trainingPermutation, testData, testPermutation, labels]=getTestAndTrainingData(obj,fractionTraining, numTargets,numModelsPerTarget)
            [obj.totModels,obj.dataFeatureValues, obj.dataLabels] = obj.extractData(numTargets,numModelsPerTarget);
            resultsPermu = randsample(1:obj.totModels, obj.totModels);
            trainingPermutation = resultsPermu(1:floor(fractionTraining*obj.totModels));
            testPermutation = resultsPermu(floor(fractionTraining*obj.totModels)+1:end);
            trainingData = obj.dataFeatureValues(trainingPermutation,:);
            testData = obj.dataFeatureValues(testPermutation,:);
            labels = obj.dataLabels;
        end
        
    end
end