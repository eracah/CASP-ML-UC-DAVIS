function [methodStruct] = readConfigFile(filename)
%reads from conifg file and makes a struct with method name and parameters
fileID = fopen(filename,'r');
method = fgetl(fileID);
parameterName = fgetl(fileID);
params = fscanf(fileID,'%f');
methodStruct = struct('method',method,'parameterName',parameterName,'params',params);


end