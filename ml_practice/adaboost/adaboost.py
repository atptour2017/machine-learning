# -*- coding: UTF-8 -*-
from numpy import *

def loadSimpData():
    	datMat = matrix([[ 1 ,  2.1],
        	[ 2. ,  1.1],
        	[ 1.3,  1. ],
        	[ 1. ,  1. ],
        	[ 2. ,  1. ]])
    	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    	return datMat,classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
	#print 'dataMatrix=',dataMatrix
	#print 'dimen=',dimen
	#print 'threshVal=',threshVal
	#print 'threshIneq=',threshIneq
    	retArray = ones((shape(dataMatrix)[0],1))
	#print 'retArray:',retArray
    	if threshIneq == 'lt':
		#print type(dataMatrix[:,dimen] <= threshVal)
		#print dataMatrix[:,dimen] <= threshVal
        	retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    	else:
        	retArray[dataMatrix[:,dimen] > threshVal] = -1.0
	#print retArray
    	return retArray

def test(dataArr):
	#dim 0, thresh 1.30, thresh ineqal: lt, the weighted error is 0.200
	stumpClassify(mat(dataArr),0,1.3,'lt')

def buildStump(dataArr,classLabels,D):	#D:weight
    	dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    	m,n = shape(dataMatrix)
    	numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    	minError = inf #init error sum, to +infinity
	#print 'minError=',minError
    	for i in range(n):#loop over all dimensions
		#print 'i=',i
        	rangeMin = dataMatrix[:,i].min(); 
		#print 'rangeMin=',rangeMin
		rangeMax = dataMatrix[:,i].max();
		#print 'rangeMax=',rangeMax
        	stepSize = (rangeMax-rangeMin)/numSteps
		#print 'stepSize=',stepSize
        	for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            		for inequal in ['lt', 'gt']: #go over less than and greater than
				#print 'j=',j
                		threshVal = (rangeMin + float(j) * stepSize)
				#print 'threshVal=',threshVal
                		predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                		#print 'predictedVals=',predictedVals
				errArr = mat(ones((m,1)))
                		errArr[predictedVals == labelMat] = 0
                		weightedError = D.T*errArr  #calc total error multiplied by D
                		#print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                		if weightedError < minError:
                    			minError = weightedError
                    			bestClasEst = predictedVals.copy()
                    			bestStump['dim'] = i
                    			bestStump['thresh'] = threshVal
                    			bestStump['ineq'] = inequal
	#print 'bestStump:',bestStump
	#print 'minError:',minError
	#print 'bestClasEst:',bestClasEst
    	return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    	weakClassArr = []
    	m = shape(dataArr)[0]
    	D = mat(ones((m,1))/m)   #init D to all equal
    	aggClassEst = mat(zeros((m,1)))
    	for i in range(numIt):
		#print 'i=',i
        	bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        	alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        	bestStump['alpha'] = alpha
        	weakClassArr.append(bestStump)                  #store Stump Params in Array
        	#print "classEst: ",classEst.T
        	expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        	#print 'expon:',expon
		#print 'exp(expon):',exp(expon)
		D = multiply(D,exp(expon))                              #Calc New D for next iteration
        	#print 'D:',D
		D = D/D.sum()
		#print 'D:',D
        	#calc training error of all classifiers, if this is 0 quit for loop early (use break)
        	aggClassEst += alpha*classEst
        	#print "aggClassEst: ",aggClassEst.T
        	aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
		#print 'aggErrors=',aggErrors
        	errorRate = aggErrors.sum()/m
        	print "total error: ",errorRate
        	if errorRate == 0.0: break
    	return weakClassArr

def adaClassify(datToClass,classifierArr):
    	dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    	#print dataMatrix
	m = shape(dataMatrix)[0]
	#print m
    	aggClassEst = mat(zeros((m,1)))
    	for i in range(len(classifierArr)):
        	classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
			classifierArr[i]['thresh'],\
			classifierArr[i]['ineq'])#call stump classify
		#print classEst
		#print classifierArr[i]['alpha']
        	aggClassEst += classifierArr[i]['alpha']*classEst
        	#print 'aggClassEst:',aggClassEst
	#print 'sign(aggClassEst):',sign(aggClassEst)
    	return sign(aggClassEst)

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    	numFeat = len(open(fileName).readline().split('\t')) #get number of fields
    	dataMat = []; labelMat = []
    	fr = open(fileName)
    	for line in fr.readlines():
        	lineArr =[]
        	curLine = line.strip().split('\t')
        	for i in range(numFeat-1):
            		lineArr.append(float(curLine[i]))
        	dataMat.append(lineArr)
        	labelMat.append(float(curLine[-1]))
    	return dataMat,labelMat

def trySimpleClassify():
	datMat,classLabels=loadSimpData()
        #test(datMat)
        #print datMat
        #print classLabels
        D=mat(ones((5,1))/5)
        #buildStump(datMat,classLabels,D)
        classifierArr=adaBoostTrainDS(datMat,classLabels,9)
        print classifierArr
        adaClassify([100,100],classifierArr)

if __name__=='__main__':
	datArr,labelArr=loadDataSet('horseColicTraining2.txt')
	classifierArray=adaBoostTrainDS(datArr,labelArr,10000)
	
	testArr,testLabelArr=loadDataSet('horseColicTest2.txt')
	prediction10=adaClassify(testArr,classifierArray)
	errArr=mat(ones((67,1)))
	print 'error sum:',errArr[prediction10!=mat(testLabelArr).T].sum()
	print 'error rate=',float(errArr[prediction10!=mat(testLabelArr).T].sum())/float(67)
