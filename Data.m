%Evan Racah
%5/1/14
%Data class for extracting the data
classdef Data
    properties
        outputIndx
        startFeatureIndx
        endFeatureIndx
        targets %cell array of various targets
        numTargets
        featureNames  %1x42 cell array of string names of features
        dataFeatureValues %matrix of nmodels x nfeatures with all data
        dataLabels %nmodels x 1 with label for each target
        numFeatures
        totalModels
        
    end
    methods
        function obj=Data(Directory,DotMatFileName, outputName, startFeature, endFeature)
            % DATA. Constructor
            % Loads dotmat file into memory and extracts targetsData.
            
            f = load([Directory '/' DotMatFileName]); %makes a struct with all variable(s) stored 
            %in dot mat file (just one variable in this case, an ExperimentData object)
            
            %gets ExperimentData object from the struct (ExperimentData object
            %has same name as the dot mat file minus the '.mat')
            expDataObject = eval(['f.' DotMatFileName(1:end-length('.mat'))]); 
            obj.targets = expDataObject.targetsData; %targetsData is cell array of TargetData objects
            obj.numTargets = length(obj.targets);
            
            %get field indices
            testFields = obj.targets{1}.fields;
            indices = find(ismember(testFields,{outputName;startFeature; endFeature})); %finds index for various features
            obj.outputIndx = indices(1);
            obj.startFeatureIndx = indices(2);
            obj.endFeatureIndx = indices(3);
            
            
            %get number of features
            obj.numFeatures = obj.endFeatureIndx - obj.startFeatureIndx + 1;
            [obj.totalModels,obj.dataFeatureValues, obj.dataLabels] = obj.extractData();
            
        end
        
%         function [outputIndex, startIndex, endIndex] = getFieldIndices(outputName, startName, endName)
%             testFields = obj.targets{1}.fields;
%             indices = find(ismember(testFields,{outputName;startFeature; endFeature})); %finds index for various features
%             outputIndex = indices(1);
%             startIndex= indices(2);
%             endIndex = indices(3);
            
        function [totModels, dataFeatureValues, dataLabels]=extractData(obj) %each target has varying number of models but for now we will take the first k models of each target where k = numModelsPerTarget
            
            %gets estimate of number of models (numTargets * 2 * one of target's number of models) 
%             models = deal(models)
            maxModels = obj.numTargets* 2 *obj.targets{1}.nModels;
        
            %creates parallel matrix-array pair of features and labels
            dataFeatureValues = zeros(maxModels,obj.numFeatures);
            dataLabels = zeros(maxModels,1);
            
            %loops thru and grabs all the models for each target
            count = 1;
            for i = 1:obj.numTargets
                nModels = obj.targets{i}.nModels;
                k = count: count + nModels -1;
                dataFeatureValues(k,:) = obj.targets{i}.values(:,obj.startFeatureIndx:obj.endFeatureIndx);
                dataLabels(k) = obj.targets{i}.values(:,obj.outputIndx);
                count = count + nModels;
            end
            totModels = count-1;
            %get rid of extra zeros
            dataFeatureValues(totModels+1:end,:) = [];
            dataLabels(totModels+1:end,:) = [];
            %takes zscore of values
            %dataFeatureValues = zscore(dataFeatureValues);
            
           
        end
        

        function [trainingData,trainingPermutation, testData, testPermutation, labels]=getTestAndTrainingData(obj,fractionTraining)
            %get random permutation indices
            resultsPermu = randsample(1:obj.totalModels, obj.totalModels);
            trainingPermutation = resultsPermu(1:floor(fractionTraining*obj.totalModels));
            testPermutation = resultsPermu(floor(fractionTraining*obj.totalModels)+1:end);
            
            %gets random perumtation of data
            trainingData = obj.dataFeatureValues(trainingPermutation,:);
            testData = obj.dataFeatureValues(testPermutation,:);
            labels = obj.dataLabels;
        end
        
    end
    end
    