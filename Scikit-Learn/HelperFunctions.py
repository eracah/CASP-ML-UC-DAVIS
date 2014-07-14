__author__ = 'Evan Racah'


import pickle
import numpy as np

#TODO: Have filename be read from a config file
def recall(path, filename):
    with open(path + filename, 'rb') as f:
        return pickle.load(f)


def concatenate_arrays(listOfIndices, sources):
    """take a list of arrays (called sources ie. test and train) that each contain subarrays, and concatenate
    all the rows of every subarrays into one array for each array in the list
    listOfIndices: the indices of the subarrays from each array in sources to concatenate
    sources: list of arrays that each contain subarrays
    for example if I have two arrays, test and train that each have 60 subarrays inside them
    then my destination list will have two arrays test and train that each have all the data
    from the 60 subarrays in one array.
    """


    #make as many destination arrays (0's to begin with) as source arrays
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

def num_unique(l):
    return len(set(l))

def save_object(object, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

def load_object(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data