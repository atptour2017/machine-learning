from numpy import *

def loadDataSet():
	dataMat = []; labelMat = []
	fr = open('testSet.txt')
    	for line in fr.readlines():
        	lineArr = line.strip().split()
        	dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        	labelMat.append(int(lineArr[2]))
    	return dataMat,labelMat

def sigmoid(inX):
    	return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    	dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
	m2,n2=shape(mat(classLabels))
	print "shape(mat(classLabels)):%d,%d" % (m2,n2)
    	labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    	m,n = shape(dataMatrix)
	print "shape(datamatrix):%d,%d" % (m,n)
	m1,n1 = shape(labelMat)
	print "shape(labelMat):%d,%d" % (m1,n1)
    	alpha = 0.001
    	maxCycles = 500
    	weights = ones((n,1))
	print 'weights:',weights
    	for k in range(maxCycles):              #heavy on matrix operations
		print 'k=',k
		#print 'shape:dataMatrix*weights=',shape(dataMatrix*weights)
		#print dataMatrix*weights
        	h = sigmoid(dataMatrix*weights)     #matrix mult
		#print 'h=',h
        	error = (labelMat - h)              #vector subtraction
		#print 'error=',error
        	weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
		print 'weights=',weights
		#break
    	return weights

def stocGradAscent0(dataMatrix, classLabels):
	m,n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)   #initialize to all ones
    	for i in range(m):
		print i
        	h = sigmoid(sum(dataMatrix[i]*weights))
        	error = classLabels[i] - h
        	weights = weights + alpha * error * dataMatrix[i]
    	return weights

###run stocGradAscent0 for 200 times
def stocGradAscent01(dataMatrix, classLabels):
        m,n = shape(dataMatrix)
        alpha = 0.01
        weights = ones(n)   #initialize to all ones
	for j in range(200):
        	for i in range(m):
                	#print i
                	h = sigmoid(sum(dataMatrix[i]*weights))
                	error = classLabels[i] - h
                	weights = weights + alpha * error * dataMatrix[i]
        return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    	m,n = shape(dataMatrix)
    	weights = ones(n)   #initialize to all ones
    	for j in range(numIter):
		print 'j=',j
        	dataIndex = range(m)
        	for i in range(m):
            		alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
			#print 'alpha=',alpha
            		randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
			#print 'randIndex=',randIndex
            		h = sigmoid(sum(dataMatrix[randIndex]*weights))
            		error = classLabels[randIndex] - h
            		weights = weights + alpha * error * dataMatrix[randIndex]
            		del(dataIndex[randIndex])
		print weights
		break
    	return weights

def classifyVector(inX, weights):
    	prob = sigmoid(sum(inX*weights))
    	if prob > 0.5: return 1.0
    	else: return 0.0

def colicTest():
    	frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    	trainingSet = []; trainingLabels = []
    	for line in frTrain.readlines():
        	currLine = line.strip().split('\t')
        	lineArr =[]
        	for i in range(21):
            		lineArr.append(float(currLine[i]))
        	trainingSet.append(lineArr)
        	trainingLabels.append(float(currLine[21]))
	#print 'trainingSet:',trainingSet
	#print 'trainingLabels:',trainingLabels
    	trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    	errorCount = 0; numTestVec = 0.0
    	for line in frTest.readlines():
        	numTestVec += 1.0
        	currLine = line.strip().split('\t')
        	lineArr =[]
        	for i in range(21):
            		lineArr.append(float(currLine[i]))
        	if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            		errorCount += 1
    	errorRate = (float(errorCount)/numTestVec)
    	print "the error rate of this test is: %f" % errorRate
    	return errorRate

def multiTest():
    	numTests = 10; errorSum=0.0
    	for k in range(numTests):
        	errorSum += colicTest()
    	print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))

if __name__ == '__main__':
	datamat,labelmat=loadDataSet()
	#print datamat
	#print labelmat
	#print type(datamat)
	#print type(labelmat)
	gradAscent(datamat,labelmat)
	#stocGradAscent0(array(datamat), labelmat)
	#stocGradAscent1(array(datamat), labelmat,150)
	#colicTest()
	#multiTest()
