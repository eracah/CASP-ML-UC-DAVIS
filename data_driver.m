clear
clc
clear
clear

clear
clc
tic;
addpath('Results','Keasar');


dat = Data('.','CASP8_9_10_ends.mat', 'gdt_ts', 'bondEnergy', 'secondaryStructureFraction');
[trainingData,trainingPermutation, testData, testPermutation, labels] = dat.getTestAndTrainingData(0.8);
disp(toc)

% tic;
% [IDX, D] = knnsearch(trainingData,testData);
% testResults = labels(trainingPermutation(IDX)); %IDX is the indices of the training data testdata is close to, so the trainingPermutation(IDX) is the label for that closest negihbor
% [IDX, D] = knnsearch(trainingData,trainingData);
% trainingResults = labels(trainingPermutation(IDX));
% t1 = toc;
% disp(t1);

res = Results(labels);
%res.addNewResults(length(trainingResults),trainingResults, trainingPermutation, testResults, testPermutation);

tic;
regressionTrainingData = [trainingData ones(length(trainingData(:,1)),1)];
regressionTestData = [testData ones(length(testData(:,1)),1)];
beta = pinv(regressionTrainingData)*labels(trainingPermutation);
testResults = regressionTestData * beta;
trainingResults = regressionTrainingData * beta;
t2 = toc;
res.addNewResults(length(trainingResults),trainingResults, trainingPermutation, testResults, testPermutation);
