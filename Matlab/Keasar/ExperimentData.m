%DISCAUMER - This MATLAB class was written by Chen Keasar, BGU (chen.keasar@gmail.com). 
%It is free without any restrictions. Yet, please be advised that it was not tested very carefully. 
%Thus, the author takes NO responsibility to any damage that this class may cause to your computer, research, or health.  
%If you find bugs or add a new functionality please drop me a line (chen.keasar@gmail.com).

classdef ExperimentData < handle
    %A container class for TargetData objects (see the TargetData.m file). 
    %A targetData object contains information regarding multiple models of the same protein (target). 
    %It is assumed that all data was collected in a uniform procedure described in the readme field (an experiment),
    %  and that all targetTata elements use the same fields in the same order. 
    %A typical way to generate a data set is to use the static method getExperimentData(experimentDirectory), 
    %  where experimentDirectory is a directory that includes multiple TargetData files.  
    properties
        name %Name of the data set
        targetsData 
        copyrightAndReadme % Free format description of the data and terms of use. May be printed using the printReadme method.
    end
    methods
        
        %constructor
        function obj = ExperimentData(name)
            obj.name = name;
        end
                
        %Duplicate this object
        function dup = duplicate(obj,newName)
            dup = ExperimerntData(newName);
            for i = length(obj.targetsData()):-1:1
                dup.targetsData{i} = obj.targetsData{i}.duplicate(dup);
            end
        end
        
        %The data fields in this dataset
        function fields = getFields(obj)
            fields = obj.targetsData{1}.fields;
        end

        %A list of all targets
        function targetNames = getTargetNames(obj)
            targetNames = cellfun(@(X)X.targetName,obj.targetsData,'UniformOutput',false);
        end
        
        
        % Generate a new object with a copy of some fields
        % featureNames - a cell array with field names { 'field1' 'field2' ...} 
        function extracted = extract(obj,featureNames,newName)
            extracted = ExperimentData(newName);
            for i = length(obj.targetsData()):-1:1
                extracted.targetsData{i} = obj.targetsData{i}.extract(featureNames,extracted);
            end
        end
        
        
        % Generate a new object with a copy of some targets
        % targetNames - a cell array with field names { 'field1' 'field2' ...} 
        function extractedTargets = extractTargets(obj,targetNames,newName)
            extractedTargets             = ExperimentData(newName);
            allTargetNames               = obj.getTargetNames;
            indices                      = Util.getIndices(allTargetNames,targetNames);
            extractedTargets.targetsData = obj.targetsData(indices);
        end
        
        %In order to break the data set into training and test subsets,
        %assuming that you already know the training set (a cell array with field names { 'field1' 'field2' ...} 
        
        function validationSet = extractValidationSet(obj,trainingSet)
           targetNames          = obj.getTargetNames;
           trainingSetIndices   = Util.getIndices(targetNames,trainingSet);
           validationSetIndices = true(length(targetNames),1);
           validationSetIndices(trainingSetIndices)= false;
           validationSet        = targetNames(validationSetIndices);
           validationSet        = obj.extractTargets(validationSet,[obj.name 'ValidationSet']);
        end
        
        %Generates a new object with a single field (newFieldName) whose
        %values are the subtraction of two fields in the this object
        %(fieldsToSubtract - a cell array).
        function difference = subtract(obj,fieldsToSubtract,newFieldName, newName)
            difference             = ExperimrentData(newName);
            for i = length(obj.targetsData()):-1:1
                difference.targetsData{i} = obj.targetsData{i}.difference(fieldsToSubtract,newFieldName,difference);
            end
        end
            
        % Generates a new object whose fields are a union of this object fields and those of the otherExperimentData argument 
        % Both this and other objects are expected to include same targets in the same order. 
        % Within each target it is expected that the same models appear in the same order.
        function concatenated = concatenate(obj, otherExperimentData,newName)
                concatenated         = ExperimentData(newName);
                if (length(obj.targetsData) ~= length(otherExperimentData.targetsData))
                    error('This is weird');
                end
                for i = length(obj.targetsData):-1:1
                    concatenated.targetsData{i} = obj.targetsData{i}.concatenate(otherExperimentData.targetsData{i},concatenated);
                end
        end
        
        
        function filterByNumberOfModels(obj,minimalNumberOfModels)
            obj.targetsData(cellfun(@(X)X.nModels,obj.targetsData) <minimalNumberOfModels) = [];
        end
        
        function modified = applyExponentToGdtTs(obj,exponent,newName)
            modified = ExperimentData(newName);
            for i = length(obj.targetsData()):-1:1
                modified.targetsData{i} = obj.targetsData{i}.applyExponentToGdtTs(exponent);
            end
        end

        function readReadme(obj,fileName)
            readmeFid = fopen(fileName,'r');
            if (readmeFid == -1)
                error(['Cannot find file ' experimentDirectory '/readme']);
            end
            line = 'temp';
            iLine = 1;
            while (ischar(line))
                line = fgets(readmeFid);
                obj.copyrightAndReadme{iLine} = line;
                iLine = iLine+1;
            end
        end
        
        function printReadme(obj)
            for iLine = 1:length(obj.copyrightAndReadme)-1
                fprintf(obj.copyrightAndReadme{iLine});
            end
        end
        
        
        function figureH = plotFields(obj,fields,format)
            figureH = figure;
            extracted    = obj.extract(fields,'temp');
            for iTarget = length(extracted.targetsData):-1:1
                plot(extracted.targetsData{iTarget}.values(:,1),extracted.targetsData{iTarget}.values(:,2),format);
                hold on
            end
        end
        
    end
    
    
    
    methods(Static=true)
        function [experimentData, nModels]=getExperimentData(experimentDirectory)
            experimentData = ExperimentData(experimentDirectory);
            experimentData.readReadme([experimentDirectory '/readme.pdf']);
            d = dir([experimentDirectory '/*.targetData.mat']);
            nModels = zeros(length(d),1);
            for i = length(d):-1:1
                load([experimentDirectory '/' d(i).name]);
                nModels(i) = targetData.nModels;
                if (i < length(d))
                    if (targetData.differentFields(experimentData.targetsData{length(d)}))
                        error('This is weird');
                    end
                end
                targetData.experimentData = experimentData;
                experimentData.targetsData{i} = targetData;
            end
        end
    end
end
    
    
    
    
