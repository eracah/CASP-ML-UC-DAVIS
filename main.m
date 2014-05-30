clear
clc
addpath('Results','Keasar','Data','Config');

fid = fopen('Config/start');
theFormat = '%f';
startConfigs = fscanf(fid,theFormat);

numberOfFolds = startConfigs(1);
fractionTest = startConfigs(2);
arrayOfNumberOfTargets = startConfigs(3:end);

%get methods and params from config files
files = dir('Config/*.config');
numberOfMethods = length(files);
methodStructArray = cell(1,numberOfMethods);
for i = 1: numberOfMethods
    methodStructArray{i} = readConfigFile(files(i).name);
end

resultsCellArray = cell(1,numberOfMethods);
%get data
dat = Data('Keasar','CASP8_9_10_ends.mat', 'gdt_ts', 'bondEnergy', 'secondaryStructureFraction');

tic;
for methodIndex = 1: numberOfMethods
    resultsCellArray{methodIndex} = Results(methodStructArray{methodIndex}.method,methodStructArray{methodIndex}.parameterName);
    method = methodStructArray{methodIndex}.method;
    params = methodStructArray{methodIndex}.params;
    numberOfParameters = length(params);
    %get fold to use for parameter testing once per method?
    for sizeIndex = 1 : length(arrayOfNumberOfTargets)
      
        [foldStructArray testStruct testSize trainingSize] = dat.getKFoldsAndTestData(arrayOfNumberOfTargets(sizeIndex),numberOfFolds, fractionTest);
        foldToLeaveOut = randi(numberOfFolds,1);
        [trainingStruct, CVStruct] = leaveOneOut(foldStructArray,foldToLeaveOut,trainingSize,dat.numFeatures);
        ErrorArray = zeros(1,numberOfParameters);
        %just finds the best parameter based on the first size of training
        %data and then the rest use that parameter
        if (sizeIndex ==1)
            for paramIndex = 1:numberOfParameters
                param = params(paramIndex);
                guess = runMLMethod(method,param,trainingStruct,CVStruct);
                ErrorArray(paramIndex) = getError(guess,CVStruct.labels);
            end
        
            bestParam = params(ErrorArray == min(ErrorArray));
        end
        %passing 0 to function leaves no folds out, so the second return value
        %if an empty array (junk(
        [trainingStruct, junk] = leaveOneOut(foldStructArray,0,trainingSize,dat.numFeatures);
        trainingGuesses = runMLMethod(method,bestParam,trainingStruct,trainingStruct);
        trainingError = getError(trainingGuesses,trainingStruct.labels);
        testGuesses = runMLMethod(method,bestParam,trainingStruct,testStruct);
        testError = getError(testGuesses,testStruct.labels);
        resultsCellArray{methodIndex} = resultsCellArray{methodIndex}.addNewResults(arrayOfNumberOfTargets(sizeIndex), trainingError, trainingGuesses, testError, testGuesses, bestParam);
        
    end
    %so we have a different figure for each method
    figure(methodIndex)
    resultsCellArray{methodIndex}.visualize();
end
toc;


