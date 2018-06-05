from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from numpy import *
import csv
from divide_training import *

def getDataset(filename):
        file=open(filename,'r')
        reader=csv.reader(file)
        rawlist=list()
        for row in reader:
                rawlist.append(row)
        dataset=array(rawlist)
        return rawlist,dataset

def GaussianBayes(X,Y):
	clf = GaussianNB()
	clf.fit(X, Y)
	print clf.predict([[6.9,3.1,5.1,2.3]])

def MultinomialBayes(X,Y):
	clf=MultinomialNB()
	clf.fit(x,y)
	print clf.predict([[1,2]])
	

if __name__=='__main__':
	rawlist,dataset=getDataset('r2.data')
	training_list,test_list=random_div(rawlist,50)
        training_dataset=array(training_list)
        test_dataset=array(test_list)
        training_X=training_dataset[:,0:6]
        training_y=training_dataset[:,6]
        test_X=test_dataset[:,0:6]
        test_y=test_dataset[:,6]

	#Gaussian(test_X,test_y)	
	MultinomialBayes(test_X,test_y)
