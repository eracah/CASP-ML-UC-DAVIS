function plotAll(resultsCellArray)
%plots every method together from results cell array
colors = {'g', 'r','c','b','k','y','m'};
clf;
figure(1)
j = 1;
strings = cell(length(resultsCellArray),1);
for i = 1:length(resultsCellArray)
    robj = resultsCellArray{i};
    %makes string saying what shape and color of plot
    colorString = sprintf('-o%s', colors{j});
    errorbar(robj.numbersOfTargets,robj.trainingErrorAverages,robj.trainingErrorStdvs,colorString);
    strings{j} = [robj.methodName 'Training'];
    hold on;
    j = j + 1;
    strings{j} = [robj.methodName 'Test'];
    colorString = sprintf('-d%s', colors{j});
    errorbar(robj.numbersOfTargets,robj.testErrorAverages,robj.testErrorStdvs,colorString);
    j = j + 1;

end
legend(strings);
xlabel('Number of Training Targets')
ylabel('Mean Squared Error')
title('Learning Curve for KNN and Random Forests')