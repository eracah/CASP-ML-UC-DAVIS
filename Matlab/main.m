clear all;
clc
clf
addpath('Results','Keasar','Data','Config');

resultsFilename = input('Results Filename: ','s');
resultsPath = ['Results/Accuracies/' resultsFilename '.mat'];

%TODO -> make a class for config reading that stores all these parameters
%-----
[numberOfFolds fractionTest startConfigs arrayOfNumberOfTargets numberOfMethods methodStructArray errorType numberOfTSetsPerSize] = configureSettings();


resultsCellArray = cell(1,numberOfMethods);

%first element is label used for learning and second and third are range of
%features used
ColumnsUsed = {'gdt_ts', 'bondEnergy', 'secondaryStructureFraction'};
pathToData = ['Keasar','/','CASP8_9_10_ends.mat'];
%-------


%get data
dat = RawData(pathToData,ColumnsUsed,fractionTest);


%TODO factor a lot of this code into MLData objects
tic;
for methodIndex = 1: numberOfMethods
    fprintf('method %s\n',methodStructArray{methodIndex}.method);
    
    %make a results object for every method used
    resultsCellArray{methodIndex} = Results(methodStructArray{methodIndex}.method,methodStructArray{methodIndex}.parameterName,errorType, arrayOfNumberOfTargets);
    
    %get method and list of parameters
    method = methodStructArray{methodIndex}.method;
    params = methodStructArray{methodIndex}.params;
    numberOfParameters = length(params);
    
    %for each training target size
    for sizeIndex = 1 : length(arrayOfNumberOfTargets)
        fprintf('\t target size %f \n',arrayOfNumberOfTargets(sizeIndex));
      
      %for each trial for a given training size
      for trainingSetIndex = 1 : numberOfTSetsPerSize
          fprintf('\t \t training set trial number %f \n',trainingSetIndex);
          
            %get the test and training data
            [foldStructArray testStruct testSize trainingSize dat] = dat.getKFoldsAndTestData(arrayOfNumberOfTargets(sizeIndex),numberOfFolds);
            
            %leave one fold out and the rest is training
            foldToLeaveOut = randi(numberOfFolds,1);
            [trainingStruct, CVStruct] = leaveOneOut(foldStructArray,foldToLeaveOut,trainingSize,dat.numFeatures);
            
            
            ErrorArray = zeros(1,numberOfParameters);
            
            %loop through all parameters and find mean error for each by
            %testing on the fold that is left out
            for paramIndex = 1:numberOfParameters
                fprintf('\t \t \t param number %f \n',paramIndex);
                param = params(paramIndex);
                guess = runMLMethod(method,param,trainingStruct,CVStruct);
                ErrorArray(paramIndex) = getError(guess,CVStruct.labels, errorType);
            end

            %TODO: save the best parameters used every time and average
            %them
            %pick the parameter that had lowest error
            bestParam = params(ErrorArray == min(ErrorArray));
            bestParam = bestParam(1);
            fprintf('\t \t got best param of %f \n',bestParam);

           
            %consolidate all training folds and trains on those and tests
            %on train
            [trainingStruct, junk] = leaveOneOut(foldStructArray,0,trainingSize,dat.numFeatures);  %passing 0 to function leaves no folds out, so the second return value is an empty array (junk)
            trainingGuesses = runMLMethod(method,bestParam,trainingStruct,trainingStruct);
            trainingError = getError(trainingGuesses,trainingStruct.labels,errorType);
            
            %tests on test data
            testGuesses = runMLMethod(method,bestParam,trainingStruct,testStruct);
            testError = getError(testGuesses,testStruct.labels,errorType);
            
            %TODO: This:
                %cvMse = crossval('mse',x,y,'predfun',regf,'Options',paroptions);
                %regf is anonymous function that is prediction function
                %regf=@(XTRAIN,ytrain,XTEST)(XTEST*regress(ytrain,XTRAIN));
                
                
            %add results to results object
            resultsCellArray{methodIndex} = resultsCellArray{methodIndex}.addNewResults(arrayOfNumberOfTargets(sizeIndex), trainingError, trainingGuesses, testError, testGuesses, bestParam);
      end
    end
    
    %reassigns resultsCellArray so new data members that visualize creates
    %can be stored
    resultsCellArray{methodIndex} = resultsCellArray{methodIndex}.prepareForVisualize();
end
%visualize(all of the methods)
toc;
plotAll(resultsCellArray);
%save the results
save(resultsPath,'resultsCellArray');



