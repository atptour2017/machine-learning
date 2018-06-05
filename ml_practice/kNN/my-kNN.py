from numpy import *

def file2matrix(filename):
    	love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    	fr = open(filename)
    	arrayOLines = fr.readlines()
    	numberOfLines = len(arrayOLines)            #get the number of lines in the file
    	returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    	classLabelVector = []                       #prepare labels return
    	index = 0
    	for line in arrayOLines:
        	line = line.strip()
        	listFromLine = line.split('\t')
        	returnMat[index,:] = listFromLine[0:3]
        	if(listFromLine[-1].isdigit()):
            		classLabelVector.append(int(listFromLine[-1]))
        	else:
            		classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        	index += 1
    	return returnMat,classLabelVector

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

if __name__=='__main__':
	datmat,labelmat=file2matrix('datingTestSet2.txt')
	newdat,range,minval=autoNorm(datmat)
	print newdat
