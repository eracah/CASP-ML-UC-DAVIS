%Evan Racah
%5/1/14
%Data class for extracting the data
classdef RawData
    
    %%%%%%%%%%   Properties  %%%%%%%%%
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
        testDataObject
    end
    
    
    
    %%%%%%%%%%%%%   METHODS %%%%%%%%%%%%%%
    methods
        function obj=RawData(PathToDotMatFile,DataColumnsUsed,fractionOfTargetsThatAreTest)
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
        function  [foldDataObjects testDataObject sizeOfTestData sizeOfTrainingData obj] = getKFoldsAndTestData(obj,numberOfTrainingTargets,numberOfFolds) 
            % GETKFOLDSANDTESTDATA Training Folds and Test Data
            % returns training data fold objects and test data objects
            
            
            %selects trainingTargetIndices, a "numberOfTrainingTargets"-sized random sample of all
            %the targets not already used for test data
            startIndex = obj.numberOfTestTargets + 1;
            trainingTargetIndices = randsample(obj.targetPermutation(startIndex:end),numberOfTrainingTargets);
            
            
            [foldDataObjects modelsCount] = obj.getTrainingFolds(numberOfFolds,trainingTargetIndices,numberOfTrainingTargets);
            
           
            %only creates test DataObject the first time this member function
            %is called; otherwise it is already a data member
            if ~obj.testDataCreated
              obj = obj.computeTestData();
              
            end
            testDataObject = obj.testDataObject;
            sizeOfTestData = length(obj.testDataObject.data(:,1)); 
            sizeOfTrainingData = modelsCount; 
            
        end
        
        function [foldDataObjects modelsCount] = getTrainingFolds(obj,numberOfFolds,trainingTargetIndices,numberOfTrainingTargets)
        %divide up the trainingIndices into clusters of indices and put
        %it each in a DataObject for that fold
            modelsCount = 0;
            foldDataObjects = cell(numberOfFolds,1);
            targetCount = 1;
            for fold = 1:numberOfFolds
                
                % get a portion of the trainingTargetIndices for each fold
                foldTargetIndices = trainingTargetIndices(targetCount : targetCount + floor(numberOfTrainingTargets / numberOfFolds) - 1);
                
                %overestimate the models in each fold
                estimatedModelsPerFold = length(foldTargetIndices)*2*obj.modelsPerTargetEstimate;
                foldDataObjects{fold} = MLData(foldTargetIndices,estimatedModelsPerFold,obj.numFeatures);
                
                %fill fold with data
                foldDataObjects{fold} = obj.addDataToDataObject(foldDataObjects{fold});
                
                %tally actual models
                modelsCount = modelsCount + length(foldDataObjects{fold}.data(:,1)); 
                
                %keep track of targets used
                targetCount = targetCount + floor(numberOfTrainingTargets / numberOfFolds);
            end
        end
        
        function [obj] = computeTestData(obj)
                modelsPerTest = length(obj.testTargetIndices)*obj.modelsPerTargetEstimate;
                obj.testDataObject = MLData(obj.testTargetIndices,modelsPerTest,obj.numFeatures);
                obj.testDataObject = obj.addDataToDataObject(obj.testDataObject);
                obj.testDataCreated = true;
        end
        
        function [aTargetsData, aTargetsLabel] = getTargetData(obj,targetIndex)
            aTargetsData = obj.targetsCellArray{targetIndex}.values(:,obj.startFeatureIndex:obj.endFeatureIndex);
            
            %get zscore of the data
            aTargetsData = zscore(aTargetsData);
            
            %no zscore of the labels
            aTargetsLabel = obj.targetsCellArray{targetIndex}.values(:,obj.outputIndex);
        end
        
         
        function [theDataObject]= addDataToDataObject(obj,theDataObject)
            count = 1;
            for targetArrayIndex = 1:length(theDataObject.targetIndices)
                    targetIndex = theDataObject.targetIndices(targetArrayIndex);
                    [aTargetsData aTargetsLabel] = obj.getTargetData(targetIndex);
                    
                    
                    
                    modelsInThisTarget = length(aTargetsData(:,1));
                    modelsRange = count:count + modelsInThisTarget - 1;
                    theDataObject.data(modelsRange,:) = aTargetsData;
                    theDataObject.labels(modelsRange) = aTargetsLabel;
                    
                    theDataObject.key(modelsRange) = targetIndex.*ones(length(modelsRange),1); %key is vector to data matrix that says which
                    %target model belongs to using target index
                    count = count + modelsInThisTarget;
                    
                    
            end
            
            %clear the zeros
            theDataObject.data(count:end,:)= [];
            theDataObject.labels(count:end) = [];
            theDataObject.key(count:end) = [];
            
            
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
     