function [testResults] = runMLMethod(trainingData,trainingPermutation,testData,labels )
    
    if method == 'knn'
        [IDX, D] = knnsearch(trainingData,testData);
        testResults = labels(trainingPermutation(IDX)); %IDX is the indices of the training data testdata is close to, 
        %so the trainingPermutation(IDX) is the label for that closest negihbor
    end
    
    if method == 'linreg'
        regressionTrainingData = [ones(length(trainingData(:,1)),1) trainingData ];
        regressionTestData = [ones(length(testData(:,1)),1) testData ];
        beta = pinv(regressionTrainingData)*labels(trainingPermutation);
        testResults = regressionTestData * beta;
    end


    

