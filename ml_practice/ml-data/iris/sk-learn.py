from sklearn.naive_bayes import GaussianNB
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
	data=[[6.9,3.1,5.1,2.3]]
	print clf.predict(data)
	print clf.predict_proba(data)
	print clf.predict_log_proba(data)

if __name__=='__main__':
	rawlist,dataset=getDataset('iris.data')
	training_list,test_list=random_div(rawlist,50)
        training_dataset=array(training_list)
        test_dataset=array(test_list)
        training_X=training_dataset[:,0:4].astype(float64)
        training_y=training_dataset[:,4]
        test_X=test_dataset[:,0:4].astype(float64)
        test_y=test_dataset[:,4]

	GaussianBayes(test_X,test_y)	
