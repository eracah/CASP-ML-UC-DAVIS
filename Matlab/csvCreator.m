%Evan Racah
clear
clc
addpath('Results','Keasar','Data','Config');

%Set features used and location of data
ColumnsUsed = {'gdt_ts', 'bondEnergy', 'secondaryStructureFraction'};
pathToData = ['Keasar','/','CASP8_9_10_ends.mat'];

fractionTest = 0.2;

%instantiate dat object
rdat = RawData(pathToData,ColumnsUsed,fractionTest);


%loop through and grab each matrix of data and save to its own csv file
for targetIndex = 1:rdat.totalTargetsInDataset
    
%aTargets Label put as last column
[aTargetsData aTargetsLabel] = rdat.getTargetData(targetIndex);
fileName = sprintf('%d.csv',targetIndex);
path = './../Scikit-Learn/Targets/';
csvwrite([path fileName],[aTargetsData aTargetsLabel]);
end