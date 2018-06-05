from numpy import *

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

def standRegres(xArr,yArr):
    	xMat = mat(xArr); yMat = mat(yArr).T
    	xTx = xMat.T*xMat
    	if linalg.det(xTx) == 0.0:
        	print "This matrix is singular, cannot do inverse"
        	return
    	ws = xTx.I * (xMat.T*yMat)
    	return ws

def plot(xArr,yArr,ws):
	import matplotlib.pyplot as plt
	fig=plt.figure()

def corelation(mat1,mat2):
	return corrcoef(mat1,mat2)
		
def getYHat(xArr):
	xMat=mat(xArr)
	yHat=xMat*ws
	return yHat

def lwlr(testPoint,xArr,yArr,k=1.0):
    	xMat = mat(xArr); yMat = mat(yArr).T
    	m = shape(xMat)[0]
    	weights = mat(eye((m)))
    	for j in range(m):                      #next 2 lines create weights matrix
        	diffMat = testPoint - xMat[j,:]     #
        	weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
		#print weights[j,j]
	print weights
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

def testlwlr(xArr,yArr):
	lwlr(xArr[0],xArr,yArr,1.0)

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

def plot1(xArr,yArr,ws):
    import matplotlib.pyplot as plt
    xMat=mat(xArr)
    yMat=mat(yArr)
    yHat=xMat*ws
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    xCopy=xMat.copy()
    xCopy.sort(0)
    yHat=xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()
	
def plot2(xArr,yArr,yHat):
    import matplotlib.pyplot as plt
    xMat=mat(xArr)
    srtInd=xMat[:,1].argsort(0)
    xSort=xMat[srtInd][:,0,:]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()

from time import sleep
import json
import urllib2
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    	sleep(10)
    	myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    	searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    	pg = urllib2.urlopen(searchURL)
    	retDict = json.loads(pg.read())
    	for i in range(len(retDict['items'])):
        	try:
            		currItem = retDict['items'][i]
            		if currItem['product']['condition'] == 'new':
                		newFlag = 1
            		else: 
				newFlag = 0
            		listOfInv = currItem['product']['inventories']
            		for item in listOfInv:
                		sellingPrice = item['price']
                		if  sellingPrice > origPrc * 0.5:
                    			print "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice)
                    			retX.append([yr, numPce, newFlag, origPrc])
                    			retY.append(sellingPrice)
        	except: 
			print 'problem with item %d' % i

def setDataCollect(retX, retY):
    	searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    	searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    	searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    	searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    	searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    	searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

def getGoogleData():
	lgX=[]
	lgY=[]
	setDataCollect(lgX,lgY)


if __name__=='__main__':
	xArr,yArr=loadDataSet('ex0.txt')
	print xArr
	ws=standRegres(xArr,yArr)
	print ws
	#yHat=getYHat(xArr)
	#print corelation(yHat.T,mat(yArr))
	#testlwlr(xArr,yArr)
	getGoogleData()
