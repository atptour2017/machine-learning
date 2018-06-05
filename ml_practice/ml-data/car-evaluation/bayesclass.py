# -*- coding: UTF-8 -*-

from divide_training import *
import csv
from numpy import *
import copy
import random

def testTab():
	print 1

def loadData(filename):		#从原始数据文件读取数据
        file=open(filename,'r')
        reader=csv.reader(file)
        rawlist=list()
        for row in reader:
                rawlist.append(row)
	file.close()
        return rawlist

def calMatConP(mat,attrlist,label): 		#计算一个矩阵里的每列的概率分布，attrlist是属性列表
	m=mat.shape[1]
	p=list()
	for i in range(m):
		posi=list()
		datlist=mat.T[i].tolist()[0]	#这个列的数据，属于这个属性
		if label=='':
			print datlist
		tmplist=attrlist[i]	#这个属性里的所有值的列表，计算P的结果按attrlist里的顺序排布
		for value in tmplist:
			if label=='':
				print 'value:',value
			num=datlist.count(value)
			if label=='':
				print num
			posibility=double(num)/double(mat.shape[0])
			posi.append(posibility)
		#print posi
		p.append(posi)
	return p

def calLabelNum(rawlist,labels):         #计算各标签下数据的数量
	labellist=mat(rawlist).T[-1].tolist()[0]
	lclist=list()
	for label in labels:
		num=labellist.count(label)
		lclist.append(num)
	return lclist

def randomSelect(datalist,number):	#随机地从列表中选取number个不同的数据，组成一个新的列表返回
	cplist=copy.deepcopy(datalist)
	selectlist=list()
	for i in range(number):
		tmp=random.choice(cplist)
		cplist.remove(tmp)
		selectlist.append(tmp)
	return selectlist,cplist

def is_num_by_except(num):
    	try:
        	float(num)
        	return True
    	except ValueError:
#        	print "%s ValueError" % num
        	return False

class Bayes(object):		#通用朴素贝叶斯分类类
	def __init__(self):
		self.attrlist=list()
		self.traindat=list()		#训练数据，双层列表
		self.trainlabel=list()		#训练标签，单层列表
		self.testdat=list()		#测试数据，双层列表
		self.testlabel=list()		#测试标签，单层列表
		self.conp=list()		#条件概率，一旦划分了训练和测试数据，则训练数据的条件概率可以计算
		self.labelp=list()		#训练数据每种标签的概率
		self.disperseTag=list()

	def getDisperse(self,checkfile):
		file=open(checkfile,'r')
        	reader=csv.reader(file)
		rawlist=list()
        	for row in reader:
                	rawlist.append(row)
        	file.close()
        	for cell in rawlist:
			if cell=='d':
				self.disperseTag='disperse'
			else:
				self.disperseTag='continuity'

	def initByCompleteFile(self,filename):
		rawlist=loadData(filename)
		lenattr=len(rawlist[0])-1
		datalist=mat(mat(rawlist).T[0:lenattr].tolist()).T.tolist()	#原始数据，包含训练数据和测试数据，双层列表
		self.createAttrList(datalist)	
		labellist=mat(rawlist).T[lenattr].tolist()[0]		#原始标签，包含训练标签和测试标签，双层列表
		self.labels=list(set(labellist))

	def normalDivData(self,filename,how_to_divide,percent):	#划分训练和测试方法1:按简单比例分类
		rawlist=loadData(filename)
		lenattr=len(rawlist[0])-1
		traindata,testdata=how_to_divide(rawlist,percent)
		self.traindat=mat(mat(traindata).T[0:lenattr].tolist()).T.tolist()
		self.trainlabel=mat(traindata).T[lenattr].tolist()[0]
		self.testdat=mat(mat(testdata).T[0:lenattr].tolist()).T.tolist()
		self.testlabel=mat(testdata).T[lenattr].tolist()[0]

	def scaleDivData(self,filename,how_to_divide,percent):	#划分训练和测试方法2：将每种标签的数据按比例分类
		rawlist=loadData(filename)
		lenattr=len(rawlist[0])-1
                labellist=mat(rawlist).T[-1].tolist()[0]
		traindata=list()
		testdata=list()
		for label in self.labels:
                        labeldatalist=list()
                        for i in range(len(rawlist)):
                                if rawlist[i][-1]==label:
                                        labeldatalist.append(rawlist[i])
			print len(labeldatalist)
                        traind,testd=how_to_divide(labeldatalist,percent)
                        traindata=traindata+traind
			testdata=testdata+testd
		self.traindat=mat(mat(traindata).T[0:lenattr].tolist()).T.tolist()
		#print mat(traindata)
                self.trainlabel=mat(traindata).T[lenattr].tolist()[0]
                self.testdat=mat(mat(testdata).T[0:lenattr].tolist()).T.tolist()
                self.testlabel=mat(testdata).T[lenattr].tolist()[0]
		#print mat(testdata)

	def minDivData(self,filename,how_to_divide,percent):	#根据最少数量的标签，来取比例的训练数据，并且其他标签取相同数据来训练
		rawlist=loadData(filename)
		traindata=list()
                testdata=list()
		lclist=calLabelNum(rawlist,self.labels)
		mincount=min(lclist)
		print lclist
		print mincount
		traincount=int(mincount*(100-percent)/100)
		for label in self.labels:
			labeldatalist=list()
			for i in range(len(rawlist)):
                                if rawlist[i][-1]==label:
                                        labeldatalist.append(rawlist[i])
			trains,tests=randomSelect(labeldatalist,traincount)
			traindata=traindata+trains
			testdata=testdata+tests
		lenattr=len(rawlist[0])-1
		self.traindat=mat(mat(traindata).T[0:lenattr].tolist()).T.tolist()
                self.trainlabel=mat(traindata).T[lenattr].tolist()[0]
                self.testdat=mat(mat(testdata).T[0:lenattr].tolist()).T.tolist()
                self.testlabel=mat(testdata).T[lenattr].tolist()[0]

	def train(self):	#训练数据
		self.calPLabel()
		self.trainNativeBayes()
                self.showConP()

	def initTrainTest(self):
		rawlist=loadData(filename)
		
	def initByTrainFile(self,trainfilename,filename):
		rawtrainlist=loadData(trainfilename)
		lenattr=len(rawtrainlist[0])-1
		self.traindat=mat(mat(rawtrainlist).T[0:lenattr].tolist()).T.tolist()
		self.trainlabel=mat(rawtrainlist).T[lenattr].tolist()[0]
		rawlist=loadData(filename)
		datalist=mat(mat(rawlist).T[0:lenattr].tolist()).T.tolist()
		labellist=mat(rawlist).T[lenattr].tolist()[0]
		self.labels=list(set(labellist))
		self.createAttrList(datalist)
		
	def showAttrList(self):
		print 'self.attrlist:'
                print self.attrlist
                print

	def generateRandomData(self,percent,orifilename,newfilename):
		rawlist=loadData(orifilename)
		traindata,testdata=random_div(rawlist,percent)
		lenattr=len(rawlist[0])-1
                self.traindat=mat(mat(traindata).T[0:lenattr].tolist()).T.tolist()
                #print self.traindat
                self.trainlabel=mat(traindata).T[lenattr].tolist()[0]
		
		file=open(newfilename,'w')
        	writer=csv.writer(file)
        	for row in traindata:
                	writer.writerow(row)
		file.close()
		
	def showTrainData(self):
		print 'self.traindat:'
		print mat(self.traindat)
		print

	def makeDecisionForTest(self):
		num_error=0
		for i in range(len(self.testdat)):
			label=self.makeDecision(self.testdat[i])
			if label=='':
				print self.testdat[i]
				break
			if label!=self.testlabel[i]:
				num_error=num_error+1
		success_rate=double(len(self.testdat)-num_error)/double(len(self.testdat))
		print 'total count:',len(self.testdat)
		print 'success count:',len(self.testdat)-num_error
		print 'error count:',num_error
		print 'success_rate:',success_rate
		return success_rate
			

	def makeDecision(self,data):
		plist=self.calPosibility(data)
		i=0
		maxp=max(plist)
		if plist.count(maxp)==1:
			index=plist.index(maxp)
			label=self.labels[index]
			print label
			return label
		else:
			print 'too many maxp to make decision!'
			return ''

	def calPosibility(self,data):
		plist=list()
		for label in self.labels:
			p=self.calPosibilityByLabel(data,label)
			plist.append(p)
		print plist
		return plist

	def calPosibilityByLabel(self,data,label):        #计算单个数据在特定标签下的概率
        	plist=list()
		labeli=self.labels.index(label)
                p=1
		print labeli
                pc=self.labelp[labeli]		#PCi
		print 'print pc begin'
                if label=='':
			print pc
                p=p*pc				#P
		i=0
                for value in data:
			#print 'value:',value
			#print 'labeli:',labeli
			#print 'i:',i
			valuei=self.attrlist[i].index(value)
			#print self.attrlist[i]
			#print 'index(value):',valuei
			if label=='':
				print 'labeli:',labeli
				print 'i:',i
				print 'valuei:',valuei
                       	pxi=self.conp[labeli][i][valuei]
                        #print pxi
                        p=p*pxi
			if label=='':
				print pxi
			i=i+1
		print 'print pc end'
           	return p

	def calPLabel(self):		#计算各标签在训练数据中的概率
                labellen=len(self.trainlabel)
		plabel=list()
		for label in self.labels:
			labelcount=self.trainlabel.count(label)
			#print labelcount
			p=double(labelcount)/double(labellen)
			plabel.append(p)
		self.labelp=plabel
		print 'self.labelp:'
		print plabel

	###当某标签的数据量在数据集中过少时，此计算很有必要，用于确定训练集的数量（在这个标签上）
	def calLabelNum(self,filename):		#计算各标签下数据的数量
		rawlist=loadData(filename)
		labellist=mat(rawlist).T[-1].tolist()[0]
		lclist=list()
		for label in self.labels:
			num=labellist.count(label)
			lclist.append(num)
		return lclist	

	def createAttrList(self,datalist):		#从原始数据里获取每个属性的所有取值列表
		datamat=mat(datalist).T
		attriblen=datamat.shape[0]
		for i in range(attriblen):
			localset=set()
			locallist=datamat[i].tolist()[0]
			localset=set(locallist)
			llist=list(localset)
			self.attrlist.append(llist)
		#print self.attrlist

	def trainNativeBayes(self):		#根据训练数据计算条件概率
		attrnum=mat(self.traindat).shape[1]	#属性数量
		pcon=list()
		for label in self.labels:
			#print 'label:',label
			labeldatalist=list()
			for i in range(len(self.trainlabel)):
				if self.trainlabel[i]==label:
					labeldatalist.append(self.traindat[i])
			if label=='':
				print label
				print mat(labeldatalist)
			#print labeldatalist
			labeldatamat=mat(labeldatalist)		#求得这个标签下的所有训练数据
			p=list()
			p=calMatConP(labeldatamat,self.attrlist,label)		#求得标签下所有属性所有值的概率
			print p
			pcon.append(p)
		self.conp=pcon
		#print 'self.conp:',self.conp
	
	def showAllInfo(self):
		print '--------------------All info--------------------'
		print 'attrlist:'
                print mat(self.attrlist)
		print 'traindat:'
                print column_stack((mat(self.traindat),mat(self.trainlabel).T))
		print 'labels:'
		print self.labels
                #self.conp
                #self.labelp
		print '--------------------All info end--------------------'
		
	def showConP(self):		#学习出的条件概率
		print '--------------------condition posibility--------------------'
                print 'conp:'
                print mat(self.conp)
                print '--------------------condition posibility end--------------------'
