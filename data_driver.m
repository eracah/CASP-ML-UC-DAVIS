clear
clc
clear
clear

clear
clc
dat = Data('.','CASP8_9_10_ends.mat', 'gdt_ts', 'bondEnergy', 'secondaryStructureFraction');
[trainingData,trainingPermutation, testData, testPermutation, labels] = dat.getTestAndTrainingData(0.8);

disp('hey')
tic;
[IDX, D] = knnsearch(trainingData,testData);
disp('done')
testResults = labels(trainingPermutation(IDX)); %IDX is the indices of the training data testdata is close to, so the trainingPermutation(IDX) is the label for that closest negihbor
[IDX, D] = knnsearch(trainingData,trainingData);
t2 = toc;
disp(t2);

trainingResults = labels(trainingPermutation(IDX));

% res = Results(labels);
% res.addNewResults(length(trainingResults),trainingResults, trainingPermutation, testResults, testPermutation);
% res.saveData('./Results','data.txt');
