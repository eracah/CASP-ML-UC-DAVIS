clear
clc
clear
clear

clear
clc
dat = Data('.','CASP8_9_10_ends.mat', 'gdt_ts', 'bondEnergy', 'secondaryStructureFraction');
[trainingData,trainingPermutation, testData, testPermutation, labels] = dat.getTestAndTrainingData(0.8);

% tic;
% [IDX, D] = knnsearch(trainingData,testData);
% testResults = labels(trainingPermutation(IDX)); %IDX is the indices of the training data testdata is close to, so the trainingPermutation(IDX) is the label for that closest negihbor
% [IDX, D] = knnsearch(trainingData,trainingData);
% trainingResults = labels(trainingPermutation(IDX));
% t1 = toc;
% disp(t1);

res = Results(labels);
% res.addNewResults(length(trainingResults),trainingResults, trainingPermutation, testResults, testPermutation);

tic;
params = labels(trainingPermutation)\trainingData;
testResults = testData * params';
trainingResults = trainingData * params';
t2 = toc;
disp(t2);
% res.addNewResults(length(trainingResults),trainingResults, trainingPermutation, testResults, testPermutation);






