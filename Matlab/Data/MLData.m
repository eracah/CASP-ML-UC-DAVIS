%Evan Racah
%6/16/14
%MLData class for training or testing data
classdef MLData
    properties
        
        targetIndices
        data
        labels
        key
        
    end %properties
    
    methods
        function obj = MLData(targetInd,modelsPerObject,numberOfFeatures)
            obj.targetIndices = targetInd;
            obj.data = zeros(modelsPerObject,numberOfFeatures);
            obj.labels = zeros(modelsPerObject,1);
            obj.key = zeros(modelsPerObject,1);
        end
        
        
    end %methods
    
end %classdef