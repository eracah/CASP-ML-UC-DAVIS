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
       
            %if you change any data member in non constructor member functions you have to return the obj 
        function  [foldStructs testStruct sizeOfTestData sizeOfTrainingData]=getKFoldsAndTestData(obj,numberOfTrainingTargets,numberOfFolds,fractionThatIsTestData) 
           
            modelsCount = 0;
            numberOfTestTargets =floor((fractionThatIsTestData /(1 - fractionThatIsTestData)) * numberOfTrainingTargets);
            numberOfTargetsUsed = numberOfTrainingTargets + numberOfTestTargets;
            
            %get a size numberOfTargets used random sample (w/o replacement) of all the
            %targets and assign the majority (1- fractionThatIsTestData) of the target indices to training
            %and the rest to testing
            targetPermutation = randsample(1:obj.totalTargetsInDataset,numberOfTargetsUsed);
            trainingTargetIndices = targetPermutation(1:numberOfTrainingTargets);
            testTargetIndices = targetPermutation(numberOfTrainingTargets + 1: end);
           
            
            %divide up the trainingIndices into clusers of indices and put
            %it each in a struct for that fold
            foldStructs = cell(numberOfFolds,1);
            count = 1;
            for i = 1:numberOfFolds
                
                foldTargetIndices = trainingTargetIndices(count : count + floor(numberOfTrainingTargets / numberOfFolds) - 1);
                estimatedModelsPerFold = length(foldTargetIndices)*2*obj.modelsPerTargetEstimate;
                foldStructs{i} = struct('targetIndices',foldTargetIndices,'data',zeros(estimatedModelsPerFold,obj.numFeatures),'labels',zeros(estimatedModelsPerFold,1),'key', zeros(estimatedModelsPerFold,1));
                count = count + floor(numberOfTrainingTargets / numberOfFolds);
            end
            
            for fold = 1:numberOfFolds
                 foldStructs{fold} = obj.addDataToStruct(foldStructs{fold});
                 modelsCount = modelsCount + length(foldStructs{fold}.data(:,1)); 
            end
            modelsPerTest = length(testTargetIndices)*obj.modelsPerTargetEstimate;
            testStruct = struct('targetIndices',testTargetIndices,'data',zeros(modelsPerTest,obj.numFeatures),'labels',zeros(modelsPerTest,1),'key', zeros(estimatedModelsPerFold,1));
            testStruct = obj.addDataToStruct(testStruct);
            sizeOfTestData = length(testStruct.data(:,1)); 
            sizeOfTrainingData = modelsCount; %this will not get changed unless data object overwritten
            
        end
        
        
        
        function [aTargetsData, aTargetsLabel] = getTargetData(obj,targetIndex)
            aTargetsData = obj.targetsCellArray{targetIndex}.values(:,obj.startFeatureIndex:obj.endFeatureIndex);
            aTargetsLabel = obj.targetsCellArray{targetIndex}.values(:,obj.outputIndex);
        end
        
         
        function [theStruct]= addDataToStruct(obj,theStruct)
            count = 1;
            for targetArrayIndex = 1:length(theStruct.targetIndices)
                    targetIndex = theStruct.targetIndices(targetArrayIndex);
                    [aTargetsData aTargetsLabel] = obj.getTargetData(targetIndex);
                    modelsInThisTarget = length(aTargetsData(:,1));
                    modelsRange = count:count + modelsInThisTarget - 1;
                    theStruct.data(modelsRange,:) = aTargetsData;
                    theStruct.labels(modelsRange) = aTargetsLabel;
                    
                    theStruct.key(modelsRange) = targetIndex.*ones(length(modelsRange),1); %key is vector to data matrix that says which
                    %target model belongs to using target index
                    count = count + modelsInThisTarget;
                    
                    
            end
            %clear the zeros
            theStruct.data(count:end,:)= [];
            theStruct.labels(count:end) = [];
            theStruct.key(count:end) = [];
            
            
        end
        

     end
    end
     