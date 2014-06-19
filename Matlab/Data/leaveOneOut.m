function [trainingStruct, CVStruct] = leaveOneOut(foldStructArray,foldLeftOutIndex,totalModels,numberOfFeatures)
%puts one fold's data in the CVStruct and the rest in the trainingStruct

if foldLeftOutIndex > 0
    
    CVStruct = foldStructArray{foldLeftOutIndex};
    numberOfCVModels = size(CVStruct.data,1);
else
    numberOfCVModels = 0;
    CVStruct = [];
end

numberOfTrainingModels = totalModels - numberOfCVModels;
trainingData = zeros(numberOfTrainingModels,numberOfFeatures);
labels = zeros(numberOfTrainingModels,1);
keys = zeros(numberOfTrainingModels,1);

count = 1;

%loops through and copies data from the remaining folds into one
%trainingStruct
for i = 1:length(foldStructArray)
    if i ~= foldLeftOutIndex
      fold = foldStructArray{i};
      modelsInThisFold = size(fold.data,1);
      modelsRange = count:count + modelsInThisFold - 1;
      trainingData(modelsRange,:) = fold.data;
      labels(modelsRange) = fold.labels;
      keys(modelsRange) = fold.key;
      count = count + modelsInThisFold;
    end
end

trainingStruct = struct('data',trainingData,'labels',labels,'key',keys);
    
end