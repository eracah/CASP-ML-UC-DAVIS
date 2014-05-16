ufunction [testResults] = runMLMethod(CVObject, method, params, testOrCV)
    
    labels = CVObject.labels
    if testOrCV == 'CV'
        trainingData = CVObject.trainingData
        testData = CVObject.CVData;
        trainingPermutation = CVObject.trainingPermutation
    else
        testData = CVObject.testData;
        trainingData = [CVObject.trainingData; CVObject.CVData]
        trainingPermutation = [CVObject.trainingPermutation;CVObject.CVPermutation];
    end
    if method == 'knn'
        [IDX, D] = knnsearch(trainingData,testData);
        testResults = labels(CVObject.trainingPermutation(IDX)); %IDX is the indices of the training data testdata is close to, 
        %so the trainingPermutation(IDX) is the label for that closest negihbor
    end
    
    if method == 'linreg'
        regressionTrainingData = [trainingData ones(length(trainingData(:,1)),1)];
        regressionTestData = [testData ones(length(testData(:,1)),1)];
        beta = pinv(regressionTrainingData)*labels(CVObject.trainingPermutation);
        testResults = regressionTestData * beta;
    end


    

