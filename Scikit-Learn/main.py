__author__ = 'Evan Racah'
import numpy as np
import pylab as pl
from sklearn import neighbors, ensemble
import glob

#get targets in numpy matrices
path = './Targets/'
targetFiles = glob.glob(path + '*.csv')
targetMatrices = len(targetFiles)*[0]

#fill in list of numpy matrices, where each matrix is a target
#last column is gdt-ts values
for targetFile in targetFiles:
   targetNumber = int(targetFile[len(path):-len('.csv')])
   targetMatrices[targetNumber-1]= np.genfromtxt(targetFile, delimiter=',')










#
#
#
#
# #read from config file for parameters
# numberOfNeighbors = 5
# n_cores = -1
#
# #get training and test data from csv
#
# # X_train, X_test, y_train, y_test = cross_validation.train_test_split(
# # iris.data, iris.target, test_size=0.4, random_state=0)
#
# methods = [neighbors.KNeighborsRegressor(numberOfNeighbors),ensemble.RandomForestRegressor(n_jobs = n_cores)]
#
# for method in methods:
#     guess = method.fit(trainingSet,trainingLabels).predict(testingSet)
#     # or method.fit(trainingSet,trainingLabels)
#     #method.score(X_test, y_test)