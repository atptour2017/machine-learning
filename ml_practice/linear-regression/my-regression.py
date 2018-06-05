from numpy import *
from divide_training import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
        numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
        dataMat = []; labelMat = []
        fr = open(fileName)
        for line in fr.readlines():
                lineArr =[]
                curLine = line.strip().split('\t')
                for i in range(numFeat):
                        lineArr.append(float(curLine[i]))
                dataMat.append(lineArr)
                labelMat.append(float(curLine[-1]))
        return dataMat,labelMat

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    	return ((yArr-yHatArr)**2).sum()

def standRegres(xArr,yArr):
        xMat = mat(xArr); yMat = mat(yArr).T
        xTx = xMat.T*xMat
        if linalg.det(xTx) == 0.0:
                print "This matrix is singular, cannot do inverse"
                return
        ws = xTx.I * (xMat.T*yMat)
        return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
        xMat = mat(xArr); yMat = mat(yArr).T
        m = shape(xMat)[0]
        weights = mat(eye((m)))
        for j in range(m):                      #next 2 lines create weights matrix
                diffMat = testPoint - xMat[j,:]     #
                weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
                #print weights[j,j]
        xTx = xMat.T * (weights * xMat)
        if linalg.det(xTx) == 0.0:
                print "This matrix is singular, cannot do inverse"
                return
        ws = xTx.I * (xMat.T * (weights * yMat))
        return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
        m = shape(testArr)[0]
        yHat = zeros(m)
        for i in range(m):
                yHat[i] = lwlr(testArr[i],xArr,yArr,k)
        return yHat

def autoNorm(dataSet):
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        print minVals
        print maxVals
        ranges = maxVals - minVals
        print ranges
        normDataSet = zeros(shape(dataSet))
        m = dataSet.shape[0]
        normDataSet = dataSet - tile(minVals, (m,1))
        normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
        return normDataSet, ranges, minVals

def ridgeRegres(xMat,yMat,lam=0.2):
        xTx = xMat.T*xMat
        denom = xTx + eye(shape(xMat)[1])*lam
        if linalg.det(denom) == 0.0:
                print "This matrix is singular, cannot do inverse"
                return
        ws = denom.I * (xMat.T*yMat)
        return ws

def testRidgeRegression1():
        xArr,yArr=loadDataSet('ex0.txt')
        ws=ridgeRegres(xArr,yArr)

def ridgeTest(xArr,yArr):
        xMat = mat(xArr); yMat=mat(yArr).T
        yMean = mean(yMat,0)
        yMat = yMat - yMean     #to eliminate X0 take mean off of Y
        #regularize X's
        xMeans = mean(xMat,0)   #calc mean then subtract it off
        xVar = var(xMat,0)      #calc variance of Xi then divide by it
        xMat = (xMat - xMeans)/xVar
        numTestPts = 30
        wMat = zeros((numTestPts,shape(xMat)[1]))
        for i in range(numTestPts):
                ws = ridgeRegres(xMat,yMat,exp(i-10))
                wMat[i,:]=ws.T
        return wMat

def calRidgeError(xArr,yArr,wmat):
	xMat=mat(xArr)
	yMat=mat(yArr).T
	print 'yMat:',yMat.T
	m=wmat.shape[0]
	print m
	datnum=xMat.shape[0]
	print datnum
	yHatMat=xMat*wmat.T
	print 'yHatMat[:,0]:',yHatMat[:,0]
	print shape(yHatMat)
	print shape(tile(yMat,(1,m)))
	yErrorMat=yHatMat-tile(yMat,(1,m))
	print yErrorMat
	ErrorSum=[]
	for line in yErrorMat.T:
		ErrorSum.append((array(line)**2).sum())
	yErrorSum=mat(ErrorSum)
	print shape(yErrorSum)

def calRidgeError1(xArr,yArr,wmat):
        xMat=mat(xArr)
        yMat=mat(yArr).T
        print 'yMat:',yMat.T
        m=wmat.shape[0]
        datnum=xMat.shape[0]
        print shape(wmat)
        print wmat[0]
        print shape(wmat.T)
        print shape(xMat)
        print shape((wmat.T)[:,0])
        print (wmat.T)[:,0]
        yHatMat=xMat*((wmat.T)[:,0])

def normData(xArr,yArr):
	xMat = mat(xArr); yMat=mat(yArr).T
        yMean = mean(yMat,0)
        yMat = yMat - yMean     #to eliminate X0 take mean off of Y
        #regularize X's
        xMeans = mean(xMat,0)   #calc mean then subtract it off
        xVar = var(xMat,0)      #calc variance of Xi then divide by it
        xMat = (xMat - xMeans)/xVar
	return xMat.tolist(),yMat.tolist()

def training_begining_ridge(datmat,labelmat,percent):                #80% traing data at begining, 20% test data at end
        traindat,testdat=begin_div(datmat,percent)
        trainlabel,testlabel=begin_div(labelmat,percent)
        wmat=ridgeTest(traindat,trainlabel)
	calRidgeError1(traindat,trainlabel,wmat)

def training_begining_standard(datmat,labelmat,percent):		#80% traing data at begining, 20% test data at end
	traindat,testdat=begin_div(datmat,percent)
	trainlabel,testlabel=begin_div(labelmat,percent)
	xArr,yArr=normData(traindat,trainlabel)
	ws=standRegres(xArr,yArr)
	print ws

def training_begining_lwlr(datmat,labelmat,percent):
	traindat,testdat=begin_div(datmat,percent)
	trainlabel,testlabel=begin_div(labelmat,percent)
	trainError=[]
	for x in range(10):
		if x==0:
			continue
		k=float(x)/10.0
		yHat=lwlrTest(traindat,traindat,trainlabel,k)
		yError=rssError(trainlabel, yHat.T)
		trainError.append(yError)
	for x in range(10):
		if x==0:
                        continue
                k=float(x)/100.0
		yHat=lwlrTest(traindat,traindat,trainlabel,k)
                yError=rssError(trainlabel, yHat.T)
                trainError.append(yError)
	print trainError

def plot1(xArr,yArr,ws):
    xMat=mat(xArr)
    yMat=mat(yArr)
    yHatArr=xMat*ws
    yError=yMat-yHatArr
    showMat=column_stack((yArr,yError))
    fig=plt.figure()
    ax=fig.add_subplot(111)
    #ax.scatter(yArr,yHatArr)
    ax.plot(showMat)
    plt.show()

if __name__=='__main__':
	datmat,labelmat=loadDataSet('abalone.txt')
	training_begining_ridge(datmat,labelmat,99)
