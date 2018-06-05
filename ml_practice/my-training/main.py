# -*- coding: UTF-8 -*-

from divide_training import *
from extract_data import *

datalist,labellist=loadDataSet('horseColicTraining.txt')
print 'len of data:',len(datalist)
print 'len of label:',len(labellist)
print 'dim of data:',len(datalist[0])

oridatalist=getDataset('horseColicTraining.txt')
print len(oridatalist)

traininglist,testlist=begin_div(oridatalist,20)
print len(testlist)
print len(traininglist)
