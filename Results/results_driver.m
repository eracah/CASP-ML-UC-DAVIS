clear;
clc;
labelledData = 100 * rand(1,50);
resultsPermu = randsample(1:50, 50);
trainingResults = 100 * rand(1,40);
testingResults = 100 * rand(1,10);
trainingPermu = resultsPermu(1:40);
testingPermu = resultsPermu(41:50);


res = Results(labelledData);
res.addNewResults(length(trainingResults),trainingResults, trainingPermu, testingResults, testingPermu);
%res.saveData('./Results','data.txt');

