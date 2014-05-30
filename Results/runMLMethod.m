function [guess] = runMLMethod(method, params,trainingDataStruct,testDataStruct)
    
    trainingData = trainingDataStruct.data;
    testData = testDataStruct.data;
    labels = trainingDataStruct.labels;
    if (strcmp('knn',method))
        IDX = knnsearch(trainingData,testData,'K',params);%IDX is numberOfTestDataPoints by K array
        %averages the gdt-ts values of the k closest neighbors 
        guess = mean(labels(IDX),2); 
       
    end
    
    if (strcmp('linreg',method))
        regressionTrainingData = [ones(length(trainingData(:,1)),1) trainingData ];
        regressionTestData = [ones(length(testData(:,1)),1) testData ];
        beta = pinv(regressionTrainingData)*labels;
        guess = regressionTestData * beta;
    end
    
    if(strcmp('decisionForest',method))
        B = TreeBagger(params,trainingData,labels,'method','regression');
        guess = B.predict(testData);
    end
        
    