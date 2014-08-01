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
    [aTargetsName, aTargetsFields, aTargetsData, aTargetsFileNames] = rdat.getLabelledTargetData(targetIndex);
    firstLetter = aTargetsName(1);
    targetNumber = str2num(aTargetsName(3:end-3));
    
    if (targetNumber > 643) | firstLetter == 'R'
        %aTargets Label put as last column
        fileName = sprintf('%s.csv',aTargetsName);
        fields = 'target,pdbFile';
        for i =1:length(aTargetsFields)
            fields = sprintf('%s,%s',fields,aTargetsFields{i}); 
        end
        fields(end+1) = ',';
        
        
        %fprintf('%s\n',fields)
        path = './Targets/';
        totalPath = [path fileName];
        fid = fopen(totalPath,'w');
        fwrite(fid,fields);
        fprintf(fid,'\n');
        fclose(fid);
        for j = 1:length(aTargetsData(:,1))
            fid = fopen(totalPath,'a');
            fprintf(fid,'%s,%s,',aTargetsName,aTargetsFileNames{j});
            fclose(fid);
            dlmwrite(totalPath,aTargetsData(j,:), '-append');
        end
        

    end
end