%Evan Racah
%5/1/14
%Data class for extracting the data
classdef Data
    properties
        outputIndex
        startFeatureIndex
        endFeatureIndex
        targetsCellArray %cell array of various targets
        totalTargetsInDataset
        numFeatures
        totalModels
        modelsPerTargetEstimate
        
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
            obj.targetsCellArray = expDataObject.targetsData; %targetsData is cell array of TargetData objects
            obj.totalTargetsInDataset = length(obj.targetsCellArray);
            
            %get field indices
            testFields = obj.targetsCellArray{1}.fields;
            indices = find(ismember(testFields,{outputName;startFeature; endFeature})); %finds index for various features
            obj.outputIndex = indices(1);
            obj.startFeatureIndex = indices(2);
            obj.endFeatureIndex = indices(3);
            
            
            %get number of features
            obj.numFeatures = obj.endFeatureIndex - obj.startFeatureIndex + 1;
            %do a best estimate of the number of models per target
            obj.modelsPerTargetEstimate = floor((obj.targetsCellArray{1}.nModels + obj.targetsCellArray{obj.totalTargetsInDataset}.nModels) / 2) ;
            obj.totalModels = 0;
            for i = 1: length(obj.targetsCellArray)
                obj.totalModels = obj.totalModels + obj.targetsCellArray{i}.nModels;
            end
            
        end
       
            
        function  [foldStructs testStruct]=getKFoldsAndTestData(obj,numberOfTrainingTargets,numberOfFolds,fractionThatIsTestData) 
           
            modelsCount = 0;
            numberOfTestTargets = (fractionThatIsTestData /(1 - fractionThatIsTestData)) * numberOfTrainingTargets;
            numberOfTargetsUsed = numberOfTrainingTargets + numberOfTestTargets;
            
            %get a size numberOfTargets used random sample (w/o replacement) of all the
            %targets and assign the majority (1- fractionThatIsTestData) of the target indices to training
            %and the rest to testing
            targetPermutation = randsample(1:obj.totalTargetsInDataset,numberOfTargetsUsed);
            trainingTargetIndices = targetPermutation(1:numberOfTrainingTargets);
            testTargetIndices = targetPermutation(numberOfTrainingTargets + 1: end);
            length(testTargetIndices)
            length(trainingTargetIndices)
            
            %divide up the trainingIndices into clusers of indices and put
            %it each in a struct for that fold
           foldStructs = cell(numberOfFolds,1);
            count = 1;
            for i = 1:numberOfFolds
                
                foldTargetIndices = trainingTargetIndices(count : count + floor(numberOfTrainingTargets / numberOfFolds) - 1);
                estimatedModelsPerFold = length(foldTargetIndices)*2*obj.modelsPerTargetEstimate;
                foldStructs{i} = struct('targetIndices',foldTargetIndices,'data',zeros(estimatedModelsPerFold,obj.numFeatures),'labels',zeros(estimatedModelsPerFold,1));
                count = count + floor(numberOfTrainingTargets / numberOfFolds);
            end
            
            for fold = 1:numberOfFolds
                 foldStructs{fold} = obj.addDataToStruct(foldStructs{fold});
                 modelsCount = modelsCount + length(foldStructs{fold}.data(:,1)); 
            end
            modelsPerTest = length(testTargetIndices)*obj.modelsPerTargetEstimate;
            testStruct = struct('targetIndices',testTargetIndices,'data',zeros(modelsPerTest,obj.numFeatures),'labels',zeros(modelsPerTest,1));
            testStruct = obj.addDataToStruct(testStruct);
            modelsCount = modelsCount + length(testStruct.data(:,1)); 
            obj.totalModels = modelsCount;
            
        end
        
        
        
        function [aTargetsData, aTargetsLabel] = getTargetData(obj,targetIndex)
            aTargetsData = obj.targetsCellArray{targetIndex}.values(:,obj.startFeatureIndex:obj.endFeatureIndex);
            aTargetsLabel = obj.targetsCellArray{targetIndex}.values(:,obj.outputIndex);
        end
        
         
        function [theStruct]= addDataToStruct(obj,theStruct)
            count = 1;
            for targetIndex = 1:length(theStruct.targetIndices)
                    [aTargetsData aTargetsLabel] = obj.getTargetData(targetIndex);
                    modelsInThisTarget = length(aTargetsData(:,1));
                    theStruct.data(count:count + modelsInThisTarget - 1,:) = aTargetsData;
                    theStruct.labels(count:count + modelsInThisTarget - 1) = aTargetsLabel;
                    count = count + modelsInThisTarget;
                    
            end
            %clear the zeros
            theStruct.data(count:end,:)= []
            theStruct.labels(count:end) = []
            
        end
        
           
        
          
            
            
            
            
            
            
            
            
            
            
            
            
            
%             %creates parallel matrix-array pair of features and labels
%             dataFeatureValues = zeros(maxModels,obj.numFeatures);  
%             dataLabels = zeros(maxModels,1);
%             
%             %loops thru and grabs all the models for each target
%             count = 1;
%             for i = 1:obj.numTargets
%                 
%                 nModels = obj.targets{i}.nModels;
%                 k = count: count + nModels -1;
%                 %if find(trainingTargets == i)
%                     
%                     
%                 
%                 %adds feature data
%                 dataFeatureValues(k,1:end) = obj.targets{i}.values(:,obj.startFeatureIndex:obj.endFeatureIndex);
%                 %adds column of target id
%                 
%                 
%                 %gets labels
%                 dataLabels(k) = obj.targets{i}.values(:,obj.outputIndex);
%                 
%                
%                 count = count + nModels;
%             end
%             totModels = count-1;
%             %get rid of extra zeros
%             dataFeatureValues(totModels+1:end,:) = [];
%             dataLabels(totModels+1:end,:) = [];
%             
%             %takes zscore of values
%             %dataFeatureValues = zscore(dataFeatureValues);
%             
%            
%         end
%         
% 
%         function [trainingData,trainingPermutation, testData, testPermutation, labels]=getTestAndTrainingData(obj,fractionTraining)
%             %get random permutation indices
%             resultsPermu = randsample(1:obj.totalModels, obj.totalModels);
%             trainingPermutation = resultsPermu(1:floor(fractionTraining*obj.totalModels));
%             testPermutation = resultsPermu(floor(fractionTraining*obj.totalModels)+1:end);
%             
%             %gets random perumtation of data
%             trainingData = obj.dataFeatureValues(trainingPermutation,:);
%             testData = obj.dataFeatureValues(testPermutation,:);
%             labels = obj.dataLabels;
%         end
%         
     end
    end
%     