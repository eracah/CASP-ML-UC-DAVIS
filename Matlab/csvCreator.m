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
    [aTargetsName, aTargetsFields, aTargetsData] = rdat.getLabelledTargetData(targetIndex);
    if ((aTargetsName(end-2:end) > 643) | (aTargetsName(1) == 'R'))
        %aTargets Label put as last column
        fileName = sprintf('%s.csv',aTargetsName);
        fields = 'target'
        for i =1:length(aTargetsFields)
            fields = sprintf('%s,%s',fields,aTargetsFields{i}); 
        end
        fields(end+1) = ',';
        
        
        fprintf('%s\n',fields)
        path = './Targets/';
        fid = fopen('hey.csv','w');
        fwrite(fid,fields);
        %csvwrite('hey.csv',aTargetsData);
        break;

    end
end