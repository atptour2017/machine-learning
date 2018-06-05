# -*- coding: UTF-8 -*-

from divide_training import *
import csv
from numpy import *
import copy
import random
from math import log
import operator
import pickle


def loadData(filename):         #从原始数据文件读取数据
        file=open(filename,'r')
        reader=csv.reader(file)
        rawlist=list()
        for row in reader:
                rawlist.append(row)
        file.close()
        return rawlist

def calLabelNum(rawlist,labels):         #计算各标签下数据的数量
        labellist=[line[-1] for line in rawlist]
        lclist=list()
        for label in labels:
                num=labellist.count(label)
                lclist.append(num)
        return lclist

def randomSelect(datalist,number):      #随机地从列表中选取number个不同的数据，组成一个新的列表返回
        cplist=copy.deepcopy(datalist)
        selectlist=list()
        for i in range(number):
                tmp=random.choice(cplist)
                cplist.remove(tmp)
                selectlist.append(tmp)
        return selectlist,cplist

def calcShannon(dataSet):	#对给定数据集计算标签的香农熵
	num=len(dataSet)
	labelCount=dict()
	for fectVec in dataSet:
		curlabel=fectVec[-1]
		if curlabel not in labelCount.keys():
			labelCount[curlabel]=0
		labelCount[curlabel]=labelCount[curlabel]+1
	shannon=0.0
	for key in labelCount:
		prob=float(labelCount[key])/num
		shannon=shannon-prob*log(prob,2)
	return shannon

def splitDataSet(dataSet,axis,value):
	retDataSet=list()
	for data in dataSet:
		if data[axis]==value:
			reducedData=data[:axis]
			reducedData.extend(data[axis+1:])
			retDataSet.append(reducedData)
	return retDataSet

def splitC45DataSet(dataSet,axis,value):
        retDataSet=list()
	if '<' in value:
		split=float(value.split(' ')[-1])
		for data in dataSet:
                	if data[axis]<split:
                        	reducedData=data[:axis]
                        	reducedData.extend(data[axis+1:])
                        	retDataSet.append(reducedData)
	elif '>' in value:
		split=float(value.split(' ')[-1])
                for data in dataSet:
                        if data[axis]>split:
                                reducedData=data[:axis]
                                reducedData.extend(data[axis+1:])
                                retDataSet.append(reducedData)  
	else:
        	for data in dataSet:
                	if data[axis]==value:
                        	reducedData=data[:axis]
                        	reducedData.extend(data[axis+1:])
                        	retDataSet.append(reducedData)
        return retDataSet

def chooseBestFeatureToSplitID3(dataSet):	#根据计算信息增益选取最佳属性
	numfeature=len(dataSet[0])-1
	baseEntropy=calcShannon(dataSet)
	bestInfoGain=0.0
	bestFeature=-1
	for i in range(numfeature):
		featurelist=[f[i] for f in dataSet]
		featureset=set(featurelist)
		newEntropy=0.0
		for feature in featureset:
			subDataSet=splitDataSet(dataSet,i,feature)
			prob=len(subDataSet)/float(len(dataSet))
			newEntropy+=prob*calcShannon(subDataSet)
		infoGain=baseEntropy-newEntropy
		if (infoGain>bestInfoGain):
			bestInfoGain=infoGain
			bestFeature=i
	return bestFeature

def chooseBestFeatureToSplitC45(dataSet,labels,disperseTag,contSplit):       #根据计算信息增益率选取最佳属性
	#print 'start to choose best feature for C4.5:'
        numfeature=len(dataSet[0])-1
        baseEntropy=calcShannon(dataSet)
        bestInfoGainRatio=0.0
        bestFeature=-1
	allfeaturelist=[]
	zeroTag=False
	for i in range(numfeature):
		if disperseTag[i]=='continuity':
                        datalist=[f[i] for f in dataSet]
                        split=contSplit[labels[i]]
			featurelist=[]
			for data in datalist:
				if data<split:
                                	featurelist.append('< '+str(contSplit[labels[i]]))
				else:
                                	featurelist.append('> '+str(contSplit[labels[i]]))
			if len(set(featurelist))==1:
				zeroTag=True
			allfeaturelist.append(featurelist)
                else:
                        featurelist=[f[i] for f in dataSet]
			if len(set(featurelist))==1:
				zeroTag=True
			allfeaturelist.append(featurelist)
	#print 'dataSet:'
	#print mat(dataSet)
	#print 'allfeaturelist:'
	#print mat(allfeaturelist)
	if zeroTag:				#IV=0，只能使用信息增益来选择属性
		#print 'IV=0, must use infoGain to choose:'
		numfeature=len(dataSet[0])-1
        	baseEntropy=calcShannon(dataSet)
        	bestInfoGain=0.0
        	bestFeature=-1
        	for i in range(numfeature):
                	featurelist=allfeaturelist[i]
                	featureset=set(featurelist)
                	newEntropy=0.0
                	for feature in featureset:
				#print feature
                        	subDataSet=splitDataSet(dataSet,i,feature)
                        	prob=len(subDataSet)/float(len(dataSet))
                        	newEntropy+=prob*calcShannon(subDataSet)
                	infoGain=baseEntropy-newEntropy
			#print infoGain
                	if (infoGain>bestInfoGain):
                        	bestInfoGain=infoGain
                        	bestFeature=i
		#print 'bestFeature:',bestFeature
        	return bestFeature
	else:						#IV!=0，可以选择信息增益率来选择属性
        	for i in range(numfeature):
			featurelist=allfeaturelist[i]
                	featureset=set(featurelist)
                	newEntropy=0.0
			newIV=0.0
                	for feature in featureset:
				#print feature
                        	subDataSet=splitC45DataSet(dataSet,i,feature)
                        	prob=len(subDataSet)/float(len(dataSet))
                        	newEntropy+=prob*calcShannon(subDataSet)
				newIV-=prob*math.log(prob,2)
				if newIV==0:
					print feature
					print mat(dataSet)
                	infoGain=baseEntropy-newEntropy
			#print 'infoGain:',infoGain
			GainRatio=infoGain/newIV
			#print 'GainRatio:',GainRatio
                	if (infoGain>bestInfoGainRatio):
                        	bestInfoGainRatio=GainRatio
                        	bestFeature=i
		#print 'bestFeature:',bestFeature
		#print 'end of choose best feature for C4.5'
        	return bestFeature

def majorityCnt(classList):
	classCount={}
	for vote in classList:
		if vote not in classCount.keys(): 
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    	return sortedClassCount[0][0]

def createID3Tree(dataSet,labels):
    	classList = [example[-1] for example in dataSet]
    	if classList.count(classList[0]) == len(classList):
        	return classList[0]		#stop splitting when all of the classes are equal
    	if len(dataSet[0]) == 1: 		#stop splitting when there are no more features in dataSet
        	return majorityCnt(classList)
    	bestFeat = chooseBestFeatureToSplitID3(dataSet)
    	bestFeatLabel = labels[bestFeat]
    	myTree = {bestFeatLabel:{}}
    	del(labels[bestFeat])
    	featValues = [example[bestFeat] for example in dataSet]
    	uniqueVals = set(featValues)
    	for value in uniqueVals:
        	subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        	myTree[bestFeatLabel][value] = createID3Tree(splitDataSet(dataSet, bestFeat, value),subLabels)
    	return myTree

def createC45Tree(dataSet,labels,disperseTag,contSplit):
	classList = [example[-1] for example in dataSet]
        if classList.count(classList[0]) == len(classList):
                return classList[0]             #stop splitting when all of the classes are equal
        if len(dataSet[0]) == 1:                #stop splitting when there are no more features in dataSet
                return majorityCnt(classList)
        bestFeat = chooseBestFeatureToSplitC45(dataSet,labels,disperseTag,contSplit)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}
        del(labels[bestFeat])
	if disperseTag[bestFeat]=='disperse':
        	featValues = [example[bestFeat] for example in dataSet]
        	uniqueVals = set(featValues)
	else:		#continuity
		lessValue='< '+str(contSplit[bestFeatLabel])
		greaterValue='> '+str(contSplit[bestFeatLabel])
		uniqueVals=[lessValue,greaterValue]
        for value in uniqueVals:
                subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
                myTree[bestFeatLabel][value] = createC45Tree(splitC45DataSet(dataSet, bestFeat, value),subLabels,disperseTag,contSplit)
	return myTree

def classify(inputTree,labels,testdata):
	firstStr=inputTree.keys()[0]
	secondDict=inputTree[firstStr]
	labelIndex=labels.index(firstStr)
	key=testdata[labelIndex]
	try:
		secondDict[key]
	except KeyError:
		print key
		print secondDict
	valueOfFeat=secondDict[key]
	#print key
	#print secondDict
	#print valueOfFeat
	if isinstance(valueOfFeat,dict):
		classLabel=classify(valueOfFeat,labels,testdata)
	else:
		classLabel=valueOfFeat
	return classLabel

def C45classify(inputTree,labels,testdata):
        firstStr=inputTree.keys()[0]
        secondDict=inputTree[firstStr]
        labelIndex=labels.index(firstStr)
        key=testdata[labelIndex]
	print key
	if isinstance(key,float):
		featlist=secondDict.keys()
		split=float(featlist[0].split(' ')[-1])
		for feat in featlist:
			if '<' in feat:
				if key<split:
					valueOfFeat=secondDict[feat]
			if '>' in feat:
				if key>split:
					valueOfFeat=secondDict[feat]
        else:
		valueOfFeat=secondDict[key]
        if isinstance(valueOfFeat,dict):
                classLabel=C45classify(valueOfFeat,labels,testdata)
        else:
                classLabel=valueOfFeat
        return classLabel

def calMidSplitData(cdata):		#根据原数据列表计算中间二分点
	retlist=list()
	for i in range(len(cdata)):
		if i+1<=len(cdata)-1:
			begin=cdata[i]
			end=cdata[i+1]
			midsplit=(begin+end)/2.0
			retlist.append(midsplit)
	return retlist

def calContinuityShannon(dataSet,attrindex,split):	#对数据集，计算连续数据的某个分界点的熵，用于确定最优分界点
	baseEntropy=calcShannon(dataSet)
	cdata=[line[attrindex] for line in dataSet]
	lessSubSet=list()
	greaterSubSet=list()
	for line in dataSet:
		if line[attrindex]>split:
			greaterSubSet.append(line)
		else:
			lessSubSet.append(line)
	newEntropy=0.0
	probless=len(lessSubSet)/float(len(dataSet))
        newEntropy+=probless*calcShannon(lessSubSet)
	probgreater=len(greaterSubSet)/float(len(dataSet))
        newEntropy+=probgreater*calcShannon(greaterSubSet)
        infoGain=baseEntropy-newEntropy
	return infoGain

def calBestSplitPoint(dataSet,attrindex):
	cdata=[line[attrindex] for line in dataSet]
	cdata.sort()
	midSplitData=calMidSplitData(cdata)
	maxGain=0
	maxSplit=0
	for split in midSplitData:
		gain=calContinuityShannon(copy.deepcopy(dataSet),attrindex,split)
		if gain>maxGain:
			maxGain=gain
			maxSplit=split
	return maxSplit	

class DecisionTree(object):	#通用决策树
	def __init__(self,attrfile):
		file=open(attrfile,'r')
		reader=csv.reader(file)
		attrlist=list()
		for line in reader:
			attrlist.append(line)
		self.traindat=list()
		self.testdat=list()
		self.attrs=attrlist[0]		#属性名列表
		self.labels=list()
		self.tree=dict()
		self.disperseTag=list()
		self.ID3Tree=dict()
		self.C45Tree=dict()
		self.contSplit=dict()		#记录每个连续属性计算出的最佳划分点：key是属性名，value是划分点的值

	def initByCompleteFile(self,filename,dispersefile):
                rawlist=loadData(filename)
                self.getDisperse(dispersefile)
                self.labels=list(set([data[-1] for data in rawlist]))

	def getDisperse(self,checkfile):        #确定某属性数据是离散还是连续。如果是连续，要转换成数字(不在本方法转换)
                file=open(checkfile,'r')
                reader=csv.reader(file)
                rawlist=list()
                for row in reader:
                        rawlist.append(row)
                file.close()
                for cell in rawlist[0]:
                        if cell=='d':
                                self.disperseTag.append('disperse')
                        else:
                                self.disperseTag.append('continuity')

	def normalDivData(self,filename,how_to_divide,percent): 		#划分训练和测试方法1:按简单比例分类
                rawlist=loadData(filename)
                lenattr=len(rawlist[0])-1
                traindata,testdata=how_to_divide(rawlist,percent)
		self.traindat=traindata
		self.testdat=testdata

	def scaleDivData(self,filename,how_to_divide,percent):  #划分训练和测试方法2：将每种标签的数据按比例分类
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
		self.traindat=traindata
		self.testdat=testdata

	def minDivData(self,filename,how_to_divide,percent):    	#根据最少数量的标签，来取比例的训练数据，并且其他标签取相同数据来训练
                rawlist=loadData(filename)
                traindata=list()
                testdata=list()
                lclist=calLabelNum(rawlist,self.labels)
		print lclist
                mincount=min(lclist)
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
		self.traindat=traindata
		print len(self.traindat)
		self.testdat=testdata
		print len(self.testdat)

	def changeContinuousData(self):         #在分了训练数据和测试数据之后，对他们各自的连续数据分别转换成数字
                #process for train dat
                rownum=len(self.traindat)
                columnnum=len(self.traindat[0])
		print self.disperseTag
                for i in range(rownum):
                        for j in range(columnnum-1):
                                if self.disperseTag[j]=='continuity':
                                        self.traindat[i][j]=float(self.traindat[i][j])
                        #print self.traindat[i]

                #process for test dat
                if len(self.testdat)==0:
                        return
                rownum=len(self.testdat)
                columnnum=len(self.testdat[0])
                for i in range(rownum):
                        for j in range(columnnum-1):
                                if self.disperseTag[j]=='continuity':
                                        self.testdat[i][j]=float(self.testdat[i][j])

	def C45Train(self):
		dataSet=copy.deepcopy(self.traindat)
                attributes=copy.deepcopy(self.attrs)
		columnnum=len(dataSet[0])
		print self.disperseTag
		for i in range(columnnum-1):
			if self.disperseTag[i]=='continuity':		#计算最佳划分点，并计算信息增益率
				maxSplit=calBestSplitPoint(dataSet,i)
				self.contSplit[self.attrs[i]]=maxSplit
		C45tree=createC45Tree(dataSet,copy.deepcopy(self.attrs),copy.deepcopy(self.disperseTag),copy.deepcopy(self.contSplit))	
		self.C45Tree=C45tree	
			

	def ID3Train(self):
		print 'len of train data:'
		print len(self.traindat)
		dataSet=copy.deepcopy(self.traindat)
		attributes=copy.deepcopy(self.attrs)
		self.ID3Tree=createID3Tree(dataSet,attributes)

	def ID3Predict(self,data):
		return classify(self.ID3Tree,self.attrs,data)

    	def ID3PredictTest(self):
        	num_error=0
        	for i in range(len(self.testdat)):
			print self.testdat[i]
            		label=self.ID3Predict(self.testdat[i])
			print label
            		if label!=self.testdat[i][-1]:
                		num_error=num_error+1
			if i!=0:
				print 'success_rate:',float(i-num_error)/i
        	success_rate=double(len(self.testdat)-num_error)/double(len(self.testdat))
        	print 'total count:',len(self.testdat)
        	print 'success count:',len(self.testdat)-num_error
        	print 'error count:',num_error
        	print 'success_rate:',success_rate
        	return success_rate

	def C45Predict(self,data):
		return C45classify(self.C45Tree,self.attrs,data)

	def storeID3Tree(self,filename):
		fw=open(filename,'w')
		pickle.dump(self.ID3Tree,fw)
		fw.close()

	def storeC45Tree(self,filename):
		fw=open(filename,'w')
                pickle.dump(self.C45Tree,fw)
                fw.close()

	def restoreID3Tree(self,filename):
		fr=open(filename)
		self.ID3Tree=pickle.load(fr)

	def restoreC45Tree(self):
		fr=open(filename)
                self.C45Tree=pickle.load(fr)

	def showAllInfo(self):
                print '--------------------All info--------------------'
                print 'attributes:'
                print self.attrs
		print 'len of train data:'
		print len(self.traindat)
                print 'training data:'
                print mat(self.traindat)
		print 'len of test data:'
		print len(self.testdat)
		print 'test data:'
		print mat(self.testdat)
                print 'labels:'
                print self.labels
		print 'ID3Tree:'
		print self.ID3Tree
		print 'C4.5Tree:'
		print self.C45Tree
                print '--------------------All info end--------------------'

if __name__=='__main__':
	datalist=loadData('purexigua.data')
	labels=['seze','gendi','qiaosheng','wenli','qibu','chugan']
	bestf=chooseBestFeatureToSplit(datalist)
	mytree=createTree(datalist,copy.copy(labels))
	print mytree
	print classify(mytree,labels,['wuhei','quansuo','chenmen','qingxi','aoxian','yinghua'])
