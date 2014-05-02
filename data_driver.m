clear
clc
clear
clear
dotMat = load('CASP8_9_10_ends.mat');
dat = Data(dotMat.CASP8_9_10_ends, 'gdt_ts', 'bondEnergy', 'secondaryStructureFraction');
[trainingData,trainingPermutation, testData, testPermutation, labels] = dat.getTestAndTrainingData(0.8,20,10)
[IDX, D] = knnsearch(trainingData,testData);
testResults = labels(trainingPermutation(IDX)); %IDX is the indices of the training data testdata is close to, so the trainingPermutation(IDX) is the label for that closest negihbor
[IDX, D] = knnsearch(trainingData,trainingData)
trainingResults = labels(trainingPermutation(IDX))

res = Results(labels);
res.addNewResults(length(trainingResults),trainingResults, trainingPermutation, testResults, testPermutation);
%res.saveData('./Results','data.txt');
