clear
clc
clear
clear

clear
clc
tic;
addpath('Results','Keasar','Data');


dat = Data('.','CASP8_9_10_ends.mat', 'gdt_ts', 'bondEnergy', 'secondaryStructureFraction');
[f g] = dat.getKFoldsAndTestData(141,10,0.2);

