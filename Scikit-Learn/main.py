__author__ = 'Evan Racah'
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import glob
import time
import random


def concatenateArrays(listOfIndices, sources):
    '''take a list of arrays that each contain subarrays, sources, and concatenates the subarrays into one array
    for each array in the list
    listOfIndices: the indices of the subarrays from each array in sources to concatenate
    sources: list of arrays that each contain subarrays
    '''

    #make as many destination arrays as source arrays
    destinations = [0 for _ in sources]

    #loop through the listOfIndices
    for count, targetIndex in enumerate(listOfIndices):

        #if we're on the first iteration set an initial value for the destinations (replace the 0's)
        if count == 0:
            #for each source make the destination equal to the targetIndex-th subarray from that source
            for Index in range(len(sources)):
                destinations[Index] = sources[Index][targetIndex]

        else:
            #for each source concatenate the destination and the targetIndex-th subarray from that source
            for Index in range(len(destinations)):
                destinations[Index] = np.concatenate((destinations[Index], sources[Index][targetIndex]))
    return destinations




#get list of target files
path = './Targets/'
targetFiles = glob.glob(path + '*.csv')

#TODO: get these from config files
test_size = 0.2
trainingSizes = [#5,
                 10]
''',
                 20,
                 50,
                 100,
                 150,
                 200,
                 230]'''
estimators = [KNeighborsRegressor(), RandomForestRegressor()]
parameterGrids =[{'n_neighbors': [1, 2, 3, 5, 9]}, {'n_estimators': [1, 2, 4, 8, 16 ]}]
scoring = 'mean_squared_error'
n_folds = 5

#preallocate python lists
inputMatrices = len(targetFiles)*[0]
outputMatrices = len(targetFiles)*[0]

#convert every target csv file to numpy matrix and split up between
#input and output (label)
for targetFile in targetFiles:
    targetNumber = int(targetFile[len(path):-len('.csv')])
    target = np.genfromtxt(targetFile, delimiter=',')
    inputMatrices[targetNumber-1] = target[:, 0:-1]
    outputMatrices[targetNumber-1] = target[:, -1]


#convert python list to numpy array (inefficient?) (needed?)
inputMatrices = np.asarray(inputMatrices)
outputMatrices = np.asarray(outputMatrices)

#split up into training and testing using test_size as the proportion value
xTotalTrain, xTotalTest, yTotalTrain, yTotalTest = train_test_split(inputMatrices, outputMatrices, test_size=test_size)

#concatenate all the targets from test together into one big array (x for features, y for labels)
xTest, yTest = concatenateArrays(range(len(xTotalTest)), [xTotalTest, yTotalTest])




for trainingSize in trainingSizes:
    #get correctly sized subset of training data
    trainIndices = random.sample(range(len(xTotalTrain)), trainingSize)

    #put all targets selected above into one array per x and y
    xTrain, yTrain = concatenateArrays(trainIndices, [xTotalTrain, yTotalTrain])

    #for each estimator (ML technique)
    for index, estimator in enumerate(estimators):
        #instantiate grid search object
        grid_search = GridSearchCV(estimator, parameterGrids[index], scoring=scoring, n_jobs=1, cv=n_folds)

        print("Performing grid search...")
        print("parameters:")

        t0 = time.time()

        #find best fit for training data
        grid_search.fit(xTrain, yTrain)
        print("done in %0.3fs" % (time.time() - t0))

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        print best_parameters

        print grid_search.score(xTest, yTest)
        print grid_search.grid_scores_

        print grid_search.best_estimator_.predict(xTest)
        print metrics.mean_squared_error(grid_search.best_estimator_.predict(xTest), yTest)














