from math import log
import csv
import numpy as np
from divide_training import *
import operator
import copy

def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
    	labelCounts = {}
    	for featVec in dataSet: #the the number of unique elements and their occurance
        	currentLabel = featVec[-1]
        	if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        	labelCounts[currentLabel] += 1
    	shannonEnt = 0.0
    	for key in labelCounts:
        	prob = float(labelCounts[key])/numEntries
        	shannonEnt -= prob * log(prob,2) #log base 2
    	return shannonEnt

def splitDataSet(dataSet, axis, value):
    	retDataSet = []
    	for featVec in dataSet:
        	if featVec[axis] == value:
            		reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            		reducedFeatVec.extend(featVec[axis+1:])
            		retDataSet.append(reducedFeatVec)
    	return retDataSet

def chooseBestFeatureToSplit(dataSet):
    	numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    	baseEntropy = calcShannonEnt(dataSet)
    	bestInfoGain = 0.0; bestFeature = -1
    	for i in range(numFeatures):        #iterate over all the features
        	featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        	uniqueVals = set(featList)       #get a set of unique values
        	newEntropy = 0.0
        	for value in uniqueVals:
            		subDataSet = splitDataSet(dataSet, i, value)
            		prob = len(subDataSet)/float(len(dataSet))
            		newEntropy += prob * calcShannonEnt(subDataSet)
        	infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        	if (infoGain > bestInfoGain):       #compare this to the best gain so far
            		bestInfoGain = infoGain         #if better than current best, set to best
            		bestFeature = i
    	return bestFeature                      #returns an integer

def majorityCnt(classList):
    	classCount={}
    	for vote in classList:
        	if vote not in classCount.keys(): classCount[vote] = 0
        	classCount[vote] += 1
    	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    	return sortedClassCount[0][0]

def createTree(dataSet,labels):
    	classList = [example[-1] for example in dataSet]
    	if classList.count(classList[0]) == len(classList):
        	return classList[0]#stop splitting when all of the classes are equal
    	if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        	return majorityCnt(classList)
    	bestFeat = chooseBestFeatureToSplit(dataSet)
    	bestFeatLabel = labels[bestFeat]
    	myTree = {bestFeatLabel:{}}
    	del(labels[bestFeat])
    	featValues = [example[bestFeat] for example in dataSet]
    	uniqueVals = set(featValues)
    	for value in uniqueVals:
        	subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        	myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    	return myTree


def classify(inputTree,featLabels,testVec):
    	firstStr = inputTree.keys()[0]
	print firstStr
    	secondDict = inputTree[firstStr]
	print featLabels
    	featIndex = featLabels.index(firstStr)
    	key = testVec[featIndex]
	print key
    	valueOfFeat = secondDict[key]
    	if isinstance(valueOfFeat, dict):
        	classLabel = classify(valueOfFeat, featLabels, testVec)
    	else: classLabel = valueOfFeat
    	return classLabel

def storeTree(inputTree,filename):
    	import pickle
    	fw = open(filename,'w')
    	pickle.dump(inputTree,fw)
    	fw.close()

def grabTree(filename):
    	import pickle
    	fr = open(filename)
    	return pickle.load(fr)

def getDataset():
        """
        import urllib
        # url with dataset
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
        # download the file
        raw_data = urllib.urlopen(url)
        dataset = np.loadtxt(raw_data, delimiter=",")
        """
        file=open('pima-indians-diabetes.data','r')
        reader=csv.reader(file)
        rawlist=list()
        for row in reader:
                rawlist.append(row)
        dataset=np.array(rawlist)
        return rawlist,dataset

def getLabels():
	"""
	1. Number of times pregnant
   	2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   	3. Diastolic blood pressure (mm Hg)
   	4. Triceps skin fold thickness (mm)
   	5. 2-Hour serum insulin (mu U/ml)
   	6. Body mass index (weight in kg/(height in m)^2)
   	7. Diabetes pedigree function
   	8. Age (years)
   	9. Class variable (0 or 1)
	"""
	labels=['Number of times pregnant','Plasma glucose','Diastolic blood pressure','Triceps skin fold thickness','2-Hour serum insulin','Body mass index','Diabetes pedigree function','Age']
	return labels

if __name__ == '__main__':
	rawlist,dataset=getDataset()
	training_list,test_list=end_div(rawlist,10)
	training_dataset=np.array(training_list)
        test_dataset=np.array(test_list)
	training_X=training_dataset[:,0:8].astype(np.float64)
        training_y=training_dataset[:,8].astype(np.int8)
	test_X=test_dataset[:,0:8].astype(np.float64)
        test_y=test_dataset[:,8].astype(np.int8)
	labels=getLabels()
	mytree=createTree(training_list,copy.copy(labels))
	print labels
	print mytree
	test_data=[1,93,70,31,0,30.4,0.315,23]	#expected 0
	result=classify(mytree,labels,test_data)
	print result
