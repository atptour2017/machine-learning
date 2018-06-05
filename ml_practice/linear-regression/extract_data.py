import csv

def loadDataSet(fileName):
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

def splitRow(row):
	text=row[0].strip()
	newrow=list()
	if ' ' in text:
		newrow=text.split(' ')
	if '\t' in text:
		newrow=text.split('\t')
	return newrow

def getDataset(filename):
        file=open(filename,'r')
        reader=csv.reader(file)
        rawlist=list()
        for row in reader:
		if len(row)==1:
			newrow=splitRow(row)
                rawlist.append(newrow)
        return rawlist

def getDataLabel(data)
	datalist=list()
	labellist=list()
	for row in datalist:
		datalist.append(row[0:-1])
		labellist.append(row[-1])
	return datalist,labellist

def extractTabFile(filename)
	pass

def extractCsvFile(filename)
	rawlist=getDataset(filename)
	traininglist,testlist=end_div(rawlist,20)
	
	
