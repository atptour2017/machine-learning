# -*- coding: UTF-8 -*-
import numpy as np
import csv
from numpy import float64
dtype=np.float64
from divide_training import *

def getDataset(filename):
	file=open(filename,'r')
	reader=csv.reader(file)
	rawlist=list()
	for row in reader:
	        rawlist.append(row)
	dataset=np.array(rawlist)
	return rawlist,dataset

def standardizeData(dataset):
	# separate the data from the target attributes
	X = dataset[:,0:8].astype(np.float64)
	y = dataset[:,8].astype(np.int8)
	#print X[0]
	from sklearn import preprocessing
	# normalize the data attributes
	normalized_X = preprocessing.normalize(X,norm='l2')
	#print normalized_X
	# standardize the data attributes
	standardized_X = preprocessing.scale(X)
	#print standardized_X
	return normalized_X,standardized_X

def getFeature(X,y):
	from sklearn import metrics
	from sklearn.ensemble import ExtraTreesClassifier
	model = ExtraTreesClassifier()
	model.fit(X, y)
	# display the relative importance of each attribute
	print(model.feature_importances_)
	
	from sklearn.feature_selection import RFE
	from sklearn.linear_model import LogisticRegression
	model = LogisticRegression()
	# create the RFE model and select 3 attributes
	rfe = RFE(model, 3)
	rfe = rfe.fit(X, y)
	# summarize the selection of the attributes
	print(rfe.support_)
	print(rfe.ranking_)

def bayes(training_X,training_y,test_X,test_y):
	print
	print '--------bayes--------'
	from sklearn import metrics
	###GaussianNB:-> Gaussian Naive Bayes 高斯朴素贝叶斯
	from sklearn.naive_bayes import MultinomialNB
	model = MultinomialNB(alpha=1)
	print training_X
	model.fit(training_X, training_y)		#训练数据，fit相当于train
	print(model)
	# make predictions
	expected = test_y
	predicted = model.predict(test_X)
	print 'test_y		',test_y
	print 'predicted:	',predicted
	accuracy = model.score(test_X, test_y)
	print 'accuracy=',accuracy
	from sklearn.metrics import accuracy_score  
	accuracy2 = accuracy_score(predicted,test_y)  
	print 'accuracy2=',accuracy2
	# summarize the fit of the model
	print(metrics.classification_report(expected, predicted))
	print(metrics.confusion_matrix(expected, predicted))

def decisionTree(training_X,training_y,test_X,test_y):
	print
	print '--------decisionTree--------'
	from sklearn import metrics
	from sklearn.tree import DecisionTreeClassifier
	# fit a CART model to the data
	model = DecisionTreeClassifier()
	model.fit(training_X, training_y)
	print(model)
	# make predictions
	expected = test_y
	predicted = model.predict(test_X)
	accuracy = model.score(test_X, test_y)
        print 'accuracy=',accuracy
        from sklearn.metrics import accuracy_score
        accuracy2 = accuracy_score(predicted,test_y)
        print 'accuracy2=',accuracy2
	# summarize the fit of the model
	print(metrics.classification_report(expected, predicted))
	print(metrics.confusion_matrix(expected, predicted))

def kNN(training_X,training_y,test_X,test_y):
	print 
	print '--------kNN--------'
	from sklearn import metrics
	from sklearn.neighbors import KNeighborsClassifier
	# fit a k-nearest neighbor model to the data
	model = KNeighborsClassifier()
	model.fit(training_X, training_y)
	print(model)
	# make predictions
	expected = test_y
	predicted = model.predict(test_X)
	accuracy = model.score(test_X, test_y)
        print 'accuracy=',accuracy
        from sklearn.metrics import accuracy_score
        accuracy2 = accuracy_score(predicted,test_y)
        print 'accuracy2=',accuracy2
	# summarize the fit of the model
	print(metrics.classification_report(expected, predicted))
	print(metrics.confusion_matrix(expected, predicted))

def SVM(training_X,training_y,test_X,test_y):
	print 
	print '--------SVM--------'
	from sklearn import metrics
	from sklearn.svm import SVC
	# fit a SVM model to the data
	model = SVC()
	model.fit(training_X, training_y)
	print(model)
	# make predictions
	expected = test_y
	predicted = model.predict(test_X)
	accuracy = model.score(test_X, test_y)
        print 'accuracy=',accuracy
        from sklearn.metrics import accuracy_score
        accuracy2 = accuracy_score(predicted,test_y)
        print 'accuracy2=',accuracy2
	# summarize the fit of the model
	#print(metrics.classification_report(expected, predicted))
	#print(metrics.confusion_matrix(expected, predicted))

def adaBoost(training_X,training_y,test_X,test_y):
        print
        print '--------adaBoost--------'
	from sklearn import metrics
	from sklearn.ensemble import GradientBoostingClassifier
	model=GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=0).fit(training_X,training_y)
	model.score(test_X, test_y)
	expected = test_y
	predicted = model.predict(test_X)
	accuracy = model.score(test_X, test_y)
        print 'accuracy=',accuracy
        from sklearn.metrics import accuracy_score
        accuracy2 = accuracy_score(predicted,test_y)
        print 'accuracy2=',accuracy2
        # summarize the fit of the model
        print(metrics.classification_report(expected, predicted))
        print(metrics.confusion_matrix(expected, predicted))

def randomForest(training_X,training_y,test_X,test_y):
        print
        print '--------randomForest--------'
	from sklearn import metrics
	from sklearn.ensemble import RandomForestClassifier
	model=RandomForestClassifier(n_estimators=20)
	model.fit(training_X, training_y)
	predicted=model.predict(test_X)
	expected=test_y
	accuracy=model.score(test_X,test_y)
        print 'accuracy=',accuracy
        from sklearn.metrics import accuracy_score
        accuracy2=accuracy_score(predicted,test_y)
        print 'accuracy2=',accuracy2
        # summarize the fit of the model
        print(metrics.classification_report(expected, predicted))
        print(metrics.confusion_matrix(expected, predicted))
	

if __name__ == '__main__':
	rawlist,raw_dataset=getDataset('car.data')
	training_list,test_list=random_div(rawlist,50)
	training_dataset=np.array(training_list)
	test_dataset=np.array(test_list)
	training_X=training_dataset[:,0:6]
	training_y=training_dataset[:,6]
	test_X=test_dataset[:,0:6]
	test_y=test_dataset[:,6]

	#normalized_X,standardized_X=standardizeData(dataset)
	#getFeature(X,y)

	bayes(training_X,training_y,test_X,test_y)
	"""
	kNN(training_X,training_y,test_X,test_y)
	SVM(training_X,training_y,test_X,test_y)
	adaBoost(training_X,training_y,test_X,test_y)
	decisionTree(training_X,training_y,test_X,test_y)
	randomForest(training_X,training_y,test_X,test_y)
	"""
