function [numberOfFolds fractionTest startConfigs arrayOfNumberOfTargets numberOfMethods methodStructArray errorType numberOfTSetsPerSize] = configureSettings()
 

fid = fopen('Config/start');
theFormat = '%f';

%gets error type and methods used
errorType = fgetl(fid);
methodString = fgetl(fid);


%pull out all desired methods from start config file
[methods{1} remainder] = strtok(methodString);
k =1;
while remainder
    k = k + 1;
    [methods{k} remainder] = strtok(remainder);
end
    
%get all the other data from start config file
startConfigs = fscanf(fid,theFormat);
numberOfFolds = startConfigs(1);
fractionTest = startConfigs(2);
numberOfTSetsPerSize = startConfigs(3);
arrayOfNumberOfTargets = startConfigs(4:end);


%get methods and params from config files
startString = 'Config/';
endString ='.config';
numberOfMethods = length(methods);
methodStructArray = cell(1,numberOfMethods);
for i = 1: numberOfMethods
    methodStructArray{i} = readConfigFile([startString methods{i} endString]);
end
end
