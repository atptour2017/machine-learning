# -*- coding: UTF-8 -*-

from divide_training import *
import csv
from numpy import *

def loadData(filename):
	file=open(filename,'r')
        reader=csv.reader(file)
        datalist=list()
	labellist=list()
        for row in reader:
                datalist.append(row[:-1])
		labellist.append(row[-1])
        return datalist,labellist

def createAttribList(datalist):
	datamat=mat(datalist).T
	attriblist=list()
	attriblen=datamat.shape[0]
	for i in range(attriblen):
		localset=set()
		locallist=datamat[i].tolist()[0]
		localset=set(locallist)
		llist=list(localset)
		#print localset
		attriblist.append(llist)
	#print attriblist
	return attriblist

def calValueSum(index,value,traindat):	#计算训练集中第index个属性的 值为value的个数	训练集行数为数据数，列数为属性数
	sumdat=0
	trainmat=mat(traindat).T
	attr_dat=trainmat[index].tolist()[0]
	for item in attr_dat:
		if item==value:
			sumdat=sumdat+1
	return sumdat


def calTotalPosibility(traindat,trainlabel,attriblist):
	plist=list()
	datlen=len(traindat)
	index=0
	for attrib in attriblist:
		i=index
		lplist=list()
		for value in attrib:
			valuesum=calValueSum(i,value,traindat)
			p=double(valuesum)/double(datlen)
			lplist.append(p)
		plist.append(lplist)
		index=index+1
	return plist

def calLabelPosibility(trainlabel):
	labelset=set(trainlabel)
	labellist=list(labelset)
	labellen=len(trainlabel)
	labelcount=dict()
	labelp=dict()
	for label in labelset:
		labelcount[label]=0
		labelp[label]=0

	for label in trainlabel:
		labelcount[label]=labelcount[label]+1
	for label in labelset:
		labelp[label]=double(labelcount[label])/double(labellen)
	return labelp

def calPosibility(data,labellist,attr_list,pdat,plabel):	#pdat:list type		plabel:dict type
	plist=list()
	for label in labellist:
		p=1
		pc=plabel[label]
		print pc
		p=p*pc
		index=0
		for item in data:
			i=index
			j=attr_list[i].index(item)
			pxi=pdat[i][j]
			print 'i:',i
			print 'j:',j
			print pxi
			p=p*pxi
			index=index+1
		plist.append(p)
	return plist	
