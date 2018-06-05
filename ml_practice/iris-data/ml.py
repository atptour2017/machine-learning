import csv
import numpy as np
from divide_training import *

file=open('iris.data','r')
reader=csv.reader(file)
rawlist=list()
for row in reader:
	rawlist.append(row)

raw_array=np.array(rawlist)
print raw_array

"""
traininglist,testlist=end_div(rawlist,50)
training_array=np.array(traininglist)
print len(training_array)

test_array=np.array(testlist)
print len(test_array)

X = raw_array[:,0:3]
y = raw_array[:,4]

from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
# standardize the data attributes
standardized_X = preprocessing.scale(X)

print normalized_X
print standardized_X
"""

