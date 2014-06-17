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
        testFraction
        numberOfTestTargets
        targetPermutation
        testTargetIndices
        testDataCreated %flag showing whether test data made yet
        testStruct
    end
    methods
        function obj=Data(PathToDotMatFile,DataColumnsUsed,fractionOfTargetsThatAreTest)
            % DATA. Constructor
            % Loads dotmat file into memory and extracts targetsData.
            %obj = Data(Directory,DotMatFileName,DataColumnsUsed) returns
            %Data object with range of feature columns specified as well as
            %ouput column
            
            obj.testFraction = fractionOfTargetsThatAreTest;
            
            %set flag initially to false
            obj.testDataCreated = false;
            
            %f is a struct with all variable(s) stored 
            %in dot mat file (just one variable in this case, an ExperimentData object)
            f = load(PathToDotMatFile); 
            
            
            %gets dotmat filename
            [DotMatFileName remainder] = strtok(PathToDotMatFile,'/');
            while remainder
                [DotMatFileName remainder] = strtok(remainder,'/');
            end
         
            
            %gets the ExperimentData object from the struct (ExperimentData object
            %has same name as the dot mat file minus the '.mat')
            expDataObject = eval(['f.' DotMatFileName(1:end-length('.mat'))]); 
            obj.targetsCellArray = expDataObject.targetsData; %targetsData is cell array of TargetData objects
            obj.totalTargetsInDataset = length(obj.targetsCellArray);
            
            %get field indices for output and the range of features
            obj = obj.getIndices(DataColumnsUsed);
            
            
            %get number of features
            obj.numFeatures = obj.endFeatureIndex - obj.startFeatureIndex + 1;
            
            %get a best estimate of the number of models per target
            obj.totalModels = 0;
            for i = 1: length(obj.targetsCellArray)
                obj.totalModels = obj.totalModels + obj.targetsCellArray{i}.nModels;
            end
            obj.modelsPerTargetEstimate = floor(obj.totalModels / length(obj.targetsCellArray)) ;
            
           
            obj.numberOfTestTargets = floor(obj.testFraction * obj.totalTargetsInDataset);
            
            %get an array that is random sampling of every target in dataset (shuffles all
            %target numbers)
            obj.targetPermutation = randsample(1:obj.totalTargetsInDataset,obj.totalTargetsInDataset);
            
            %reserves the first chunk of the array for test data
            obj.testTargetIndices = obj.targetPermutation(1:obj.numberOfTestTargets);
            
            
        end
       
            %if you change any data member in non constructor member functions you have to return the obj 
        function  [foldStructs testStruct sizeOfTestData sizeOfTrainingData] = getKFoldsAndTestData(obj,numberOfTrainingTargets,numberOfFolds) 
            % GETKFOLDSANDTESTDATA Training Folds and Test Data
            %
            modelsCount = 0;
            
            
            %selects trainingTargetIndices, a "numberOfTrainingTargets"-sized random sample of all
            %the targets not already used for test data
            startIndex = obj.numberOfTestTargets + 1;
            trainingTargetIndices = randsample (obj.targetPermutation(startIndex:end),numberOfTrainingTargets);
            
           
            
            %divide up the trainingIndices into clusters of indices and put
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
            
            %only creates test struct the first time this member function
            %is called; otherwise it is already a data member
            if ~obj.testDataCreated
                modelsPerTest = length(obj.testTargetIndices)*obj.modelsPerTargetEstimate;
                obj.testStruct = struct('targetIndices',obj.testTargetIndices,'data',zeros(modelsPerTest,obj.numFeatures),'labels',zeros(modelsPerTest,1),'key', zeros(estimatedModelsPerFold,1));
                obj.testStruct = obj.addDataToStruct(obj.testStruct);
                obj.testDataCreated = true;
            end
            sizeOfTestData = length(obj.testStruct.data(:,1)); 
            testStruct = obj.testStruct;
            sizeOfTrainingData = modelsCount; 
            
        end
        
        
        
        function [aTargetsData, aTargetsLabel] = getTargetData(obj,targetIndex)
            aTargetsData = obj.targetsCellArray{targetIndex}.values(:,obj.startFeatureIndex:obj.endFeatureIndex);
            %get zscore
            aTargetsData = zscore(aTargetsData);
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
        
        function [obj]=getIndices(obj,DataColumnsUsed)
            
            testFields = obj.targetsCellArray{1}.fields;
            indices = find(ismember(testFields,DataColumnsUsed)); %finds index for various features
            obj.outputIndex = indices(1);
            obj.startFeatureIndex = indices(2);
            obj.endFeatureIndex = indices(3);
        end
        

     end
    end
     