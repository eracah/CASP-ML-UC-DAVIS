clear all;
clc
clf
addpath('Results','Keasar','Data','Config');

[numberOfFolds fractionTest startConfigs arrayOfNumberOfTargets numberOfMethods methodStructArray errorType numberOfTSetsPerSize] = configureSettings();

theGuess = 0.8;
resultsCellArray = cell(1,numberOfMethods);

%get data
dat = Data('Keasar','CASP8_9_10_ends.mat', 'gdt_ts', 'bondEnergy', 'secondaryStructureFraction');

tic;
for methodIndex = 1: numberOfMethods
    fprintf('method %s\n',methodStructArray{methodIndex}.method);
    resultsCellArray{methodIndex} = Results(methodStructArray{methodIndex}.method,methodStructArray{methodIndex}.parameterName,errorType, arrayOfNumberOfTargets);
    method = methodStructArray{methodIndex}.method;
    params = methodStructArray{methodIndex}.params;
    numberOfParameters = length(params);
    %get fold to use for parameter testing once per method?
    for sizeIndex = 1 : length(arrayOfNumberOfTargets)
        fprintf('\t target size %f \n',arrayOfNumberOfTargets(sizeIndex));
      for trainingSetIndex = 1 : numberOfTSetsPerSize
          fprintf('\t \t training set trial number %f \n',trainingSetIndex);
            [foldStructArray testStruct testSize trainingSize] = dat.getKFoldsAndTestData(arrayOfNumberOfTargets(sizeIndex),numberOfFolds, fractionTest);
            foldToLeaveOut = randi(numberOfFolds,1);
            [trainingStruct, CVStruct] = leaveOneOut(foldStructArray,foldToLeaveOut,trainingSize,dat.numFeatures);
            ErrorArray = zeros(1,numberOfParameters);
            %just finds the best parameter based on the first size of training
            %data and then the rest use that parameter

            for paramIndex = 1:numberOfParameters
                fprintf('\t \t \t param number %f \n',paramIndex);
                param = params(paramIndex);
                guess = runMLMethod(method,param,trainingStruct,CVStruct);
                ErrorArray(paramIndex) = getError(guess,CVStruct.labels, errorType);
            end

            
            bestParam = params(ErrorArray == min(ErrorArray));
            bestParam = bestParam(1);
            fprintf('\t \t got best param of %f \n',bestParam);

            %passing 0 to function leaves no folds out, so the second return value
            %is an empty array (junk)
            [trainingStruct, junk] = leaveOneOut(foldStructArray,0,trainingSize,dat.numFeatures);
            trainingGuesses = runMLMethod(method,bestParam,trainingStruct,trainingStruct);
            trainingError = getError(trainingGuesses,trainingStruct.labels,errorType);
            testGuesses = runMLMethod(method,bestParam,trainingStruct,testStruct);
            testError = getError(testGuesses,testStruct.labels,errorType);
            resultsCellArray{methodIndex} = resultsCellArray{methodIndex}.addNewResults(arrayOfNumberOfTargets(sizeIndex), trainingError, trainingGuesses, testError, testGuesses, bestParam);
      end
    end
    
    %so we have a different figure for each method
    figure(methodIndex)
    
    %reassigns resultsCellArray so new data members that visualize creates
    %can be stored
    resultsCellArray{methodIndex} = resultsCellArray{methodIndex}.prepareForVisualize();
end
%visualize(all of the methods)
plotAll(resultsCellArray);
toc;


