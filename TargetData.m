classdef TargetData < handle
    properties
        targetName
        statistics
        fileNames
        fields
        values
        experimentData
        nModels
    end
    
    methods
        %% Constructor
        function obj = TargetData(targetName,fileNames,experimentData,fields,values)
            if (nargin == 0) 
                error('This is weird');
            end
            obj.targetName = targetName;            
            if ((nargin == 5) || (nargin == 3))
                obj.fileNames  = fileNames;
                obj.experimentData = experimentData;
                if (nargin == 5)
                    obj.fields  = fields;
                    obj.values  = values;
                    obj.nModels = size(values,1);
                end
            end
        end
        %% Column-wize methods
        function dup = duplicate(obj,experimentData)
            dup = TargetData(obj.targetName,obj.fileNames,experimentData,obj.fields,obj.values);
        end
        
        function extracted = extract(obj,featureNames,experimentData)
            extracted = TargetData(obj.targetName,obj.fileNames,experimentData); 
            allFields  = obj.fields;
            if (isempty(allFields))
                error('This is weird');
            end
            indices           = Util.getIndices(allFields,featureNames);
            extracted.fields  = featureNames;
            extracted.values  = obj.values(:,indices);
            extracted.nModels = obj.nModels;
        end
        
        function difference = subtract(obj,fieldsToSubtract,experimentData,newFieldName)
            difference        = TargetData(obj.targetName,obj.fileNames,experimentData); 
            difference.fields = newFieldName;
            indices           = Util.getIndices(obj.fields,fieldsToSubtract);
            difference.values = obj.values(:,indices(1))-obj.values(:,indices(2));
            difference.nModels = obj.nModels;
        end 
        
        function concatenated = concatenate(obj, otherTargetData,experimentData)
            concatenated         = TargetData(obj.targetName,obj.fileNames,experimentData);
            concatenated.fields  = {obj.fields{:} otherTargetData.fields{:}};
            concatenated.values  = [obj.values otherTargetData.values];
            concatenated.nModels = obj.nModels;
        end
        
        function diff = differentFields(obj,other)
            if (length(obj.fields) ~= length(other.fields))
                diff = true;
            else 
                diff = false;
                for i = 1:length(obj.fields)
                    if (~strcmp(obj.fields{i},other.fields{i}))
                        diff = true;
                    end
                end
            end
        end
        
        function modified = applyExponentToGdtTs(obj,exponent)
            index = Util.getIndices(obj.fields,{'gdt_ts'});
            modified = TargetData(obj.targetName,obj.fileNames,obj.experimentData,obj.fields,obj.values);
            modified.values(:,index) = modified.values(:,index).^exponent;
        end
    end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Static methods %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    methods(Static = true)
        function fields = getEndFields(file)    
            xmlStruct = xml2struct(file);
            children = xmlStruct.Children;
            conformations = TargetData.select(children,'end');
            conformation = conformations(1);
            attributes = conformation.Attributes;
            children   = conformation.Children;
            evenChildren = TargetData.select(children,'all');
            fields  =  {attributes.Name  evenChildren.Name};
        end
        
        function fields = getStartFields(file)
            fields = TargetData.getFields(file,'start');
        end
        
        function fields = getFields(file, flag)
            xmlStruct = xml2struct(file);
            children = xmlStruct.Children;
            conformations = TargetData.select(children,flag);
            conformation = conformations(1);
            attributes = conformation.Attributes;
            children   = conformation.Children;
            evenChildren = TargetData.select(children,'all');
            fields  =  {attributes.Name  evenChildren.Name};
        end
            
             
        % Generator
        function targetData = getTargetData(baseDir,target,fields, selectionFlag,weadFlag)
            %selection flag = start/end/all
            fields = TargetData.removeFieldsFileNamesAndValue(fields);
            targetData = TargetData(target);
            if (isdir([baseDir '/' target]))
                dirString = [baseDir '/' target '/*.xml'];
                fileList = dir(dirString);
                numberOfFiles = size(fileList,1);
                errors = cell(numberOfFiles);
                targetData.fileNames = cell(numberOfFiles,1);
                data     = cell(numberOfFiles,1);
                targetData.values   = zeros(numberOfFiles,length(fields));
                targetData.nModels  = numberOfFiles;
                for iFile =1:numberOfFiles
                    fileName = fileList(iFile).name;
                    if (isempty(strfind('.N.',fileName)))
                        disp(fileName);
                        %try
                            [modelData errorMessage] = TargetData.extractConformationsDataFromXml( [baseDir '/' target '/' fileName],fields, selectionFlag );
                            if (isempty(errorMessage{:}))
                                vs = str2double({modelData{:}{:}.value});
                                errorMessage = TargetData.checkValues(vs,fileName,fields);
                                if (isempty(errorMessage))
                                    targetData.values(iFile,:) = vs;
                                    targetData.fileNames(iFile) = {fileName};
                                    data(iFile) = modelData;
                                else
                                    errors{iFile} = errorMessage;
                                end
                            else
                                    errors{iFile} = errorMessage;
                            end
                        %catch errorMessage
                        %    disp(errorMessage.message)
                        %end
                    end
                end
                targetData.fields    = fields;
                bad = cellfun(@(X)isempty(X),targetData.fileNames);
                targetData.values(bad,:)        = [];
                targetData.fileNames = targetData.fileNames(~bad);
                targetData.nModels   = size(targetData.values,1);
 
                
                
                
                if (~isempty(targetData))
                    if (weadFlag)
                        [targetData errors] = TargetData.wead(targetData);
                    end
                    save([target '.' selectionFlag '.targetData.mat'],'targetData');
                end
                if (~isempty(errors))
                    save([target '.' selectionFlag '.errors.mat'],'errors');
                end
            end
        end
        
        function [targetData errors] = wead(targetData)
            zScores   = zscore(targetData.values);
            badValues = (zScores>6) | (zScores < -6);
            errors = {};
            while(~isempty(find(badValues,1)))
                outliers = sum(badValues,2)>0;
                errors = {errors{:} ['(outliers ' {targetData.fileNames(outliers)} ')']};
                targetData.values    = targetData.values(~outliers,:);
                targetData.nModels   = size(targetData.values,1);
                targetData.fileNames = targetData.fileNames(~outliers,:);
                zScores   = zscore(targetData.values);
                badValues = (zScores>6) | (zScores < -6);
            end
        end
        
        
        function errorMessage = checkValues(values, fileName,fields)
            nanElelents = isnan(values);
            if (~isempty(find(nanElelents,1)))
                disp(fields(nanElelents));
                errorMessage = ['NaN in ' fileName];
            else
                if (~isempty(find(isinf(values),1)))
                    errorMessage = ['Inf in ' fileName];
                else
                    errorMessage = '';
                end
            end
            
        end
        
        function [conformationsData errorMessages] = extractConformationsDataFromXml( file,fields, selectionFlag )    
            xmlStruct = xml2struct(file);
            children = xmlStruct.Children;
            conformations = TargetData.select(children,selectionFlag);
            conformationsData = cell(1,length(conformations));
            errorMessages     = cell(1,length(conformations));
            for iConf = 1:length(conformations)
                conformation = conformations(iConf);
                [conformationData errorMessage] = TargetData.getConformationData(conformation,fields,selectionFlag);
                conformationsData(iConf) = {conformationData};
                errorMessages(iConf) = {errorMessage};
                conformationsData(iConf) = {conformationsData(iConf)};
            end
        end
        
        function [conformationData errorMessage] = getConformationData(conformation,fields,selectionFlag)
            attributes = conformation.Attributes;
            children   = conformation.Children;
            evenChildren = TargetData.select(children,'all');
            childrenAttributes = [evenChildren.Attributes];
            names  =  {attributes.Name  evenChildren.Name};
            childrenValues = {attributes.Value childrenAttributes.Value};
            newNames  = cell(1,length(fields));
            newValues = cell(1,length(fields));
            fileIndex = find(strcmp('fileName',names));
            if (isempty(fileIndex))
                errorMessage = 'Cannot find file name.';
                conformationData = [];
                return;
            end
            fileName = childrenValues{fileIndex(1)};
            
            errorMessage = TargetData.check(childrenValues,selectionFlag, fileName);
            if (~isempty(errorMessage))
                conformationData = [];
                return;
            end
            for i = 1:length(fields)
                j = find(strcmp(fields{i},names));
                if (length(j)<=0)
                    errorMessage = [ 'cannot find field ' fields{i} ' in ' fileName];
                    conformationData = [];
                    return;
                end
                if  (length(j) > 1)
                    errorMessage = [errorMessage 'field ' fields{i} ' ocures more than once in ' filename]; %#ok<AGROW>
                    conformationData = [];
                    return;
                end
                newNames{i} = fields{i};
                newValues{i} = childrenValues{j};
            end
            conformationData = struct('name',newNames,'value',newValues);
        end
        
        function selected = select(array,flag)
            n = length(array);% Weird array of odd length with empty odd positions.
            if (strcmp(flag,'end'))
                selected = struct(array(n-1));
            elseif (strcmp(flag,'start'))
                selected = struct(array(2));
            elseif (strcmp(flag,'all'))
                selected = array(2:2:n);
            else
                error(['unknown flag ' flag]);
            end
        end
        
        function errorMessage = check(values,selectionFlag, fileName)
            if(~strcmp(selectionFlag,'end'))
                errorMessage = '';
            else
                if (isempty(find(strcmp('MCM_END',values),1)))
                    errorMessage = ['Cannot find MCM_END in ' fileName];
                else
                    errorMessage = '';
                end
            end
        end
        
        function fields = removeFieldsFileNamesAndValue(fields)
            fields(strcmp('fileName',fields) | strcmp('value',fields))= [];
        end
    end

    
end

