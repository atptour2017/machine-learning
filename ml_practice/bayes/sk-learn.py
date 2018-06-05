from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from numpy import *
import csv
from sklearn import metrics

def loadDataSet():
        postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
        return postingList,classVec


def createVocabList(dataSet):
        vocabSet = set([])  #create empty set
        for document in dataSet:
                vocabSet = vocabSet | set(document) #union of the two sets
        return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
        returnVec = [0]*len(vocabList)
        for word in inputSet:
                if word in vocabList:
                        returnVec[vocabList.index(word)] = 1
                else: print "the word: %s is not in my Vocabulary!" % word
        return returnVec

def getDataset(filename):
        file=open(filename,'r')
        reader=csv.reader(file)
        rawlist=list()
        for row in reader:
                rawlist.append(row)
        dataset=array(rawlist)
        return rawlist,dataset

def GaussianBayes(X,Y):
	clf = GaussianNB()
	clf.fit(X, Y)
	print clf.predict([[6.9,3.1,5.1,2.3]])

def MultinomialBayes(train_X,train_y,test_X,test_y):
	clf=MultinomialNB()
	clf.fit(train_X,train_y)

	expected = test_y
        predicted = clf.predict(test_X)
        print 'test_y           ',test_y
        print 'predicted:       ',predicted
        accuracy = clf.score(test_X, test_y)
        print 'accuracy=',accuracy
        from sklearn.metrics import accuracy_score
        accuracy2 = accuracy_score(predicted,test_y)
        print 'accuracy2=',accuracy2
        # summarize the fit of the model
        print(metrics.classification_report(expected, predicted))
        print(metrics.confusion_matrix(expected, predicted))

	"""
	data=[[]]	
	print clf.predict(data)
	print clf.predict_proba(data)
	print clf.predict_log_proba(data)
	"""

def localWords(feed1,feed0):
        import feedparser
        docList=[]; classList = []; fullText =[]
        import re
        listOfTokens = re.split(r'\W*', bigString)
        return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def bagOfWords2VecMN(vocabList, inputSet):
        returnVec = [0]*len(vocabList)
        for word in inputSet:
                if word in vocabList:
                        returnVec[vocabList.index(word)] += 1
        return returnVec

def textParse(bigString):    #input is big string, #output is word list
        import re
        listOfTokens = re.split(r'\W*', bigString)
        return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
        docList=[]; classList = []; fullText =[]
        for i in range(1,26):
                wordList = textParse(open('email/spam/%d.txt' % i).read())
                docList.append(wordList)
                fullText.extend(wordList)
                classList.append(1)
                wordList = textParse(open('email/ham/%d.txt' % i).read())
                docList.append(wordList)
                fullText.extend(wordList)
                classList.append(0)
        vocabList = createVocabList(docList)#create vocabulary
	trainingSet = range(50); testSet=[]           #create test set
        for i in range(10):
                randIndex = int(random.uniform(0,len(trainingSet)))
                testSet.append(trainingSet[randIndex])
                del(trainingSet[randIndex])
        trainMat=[]; trainClasses = []
	testMat=[]; testClasses=[]
        for docIndex in trainingSet:    #train the classifier (get probs) trainNB0
                trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
                trainClasses.append(classList[docIndex])
	print shape(trainMat)
	print shape(trainClasses)
	print trainClasses
	for docIndex in testSet:    #train the classifier (get probs) trainNB0
                testMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
                testClasses.append(classList[docIndex])
	#print mat(trainMat)
	#print trainClasses
	#print testMat
	print shape(testMat)
	print shape(testClasses)
	print testClasses
        MultinomialBayes(trainMat,trainClasses,testMat,testClasses)

def simpleTest():
	listOPosts,listClasses=loadDataSet()
        #print listOPosts
        myVocabList = createVocabList(listOPosts)
        #print myVocabList
        trainMat=[]
        for postinDoc in listOPosts:
                trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
        print mat(trainMat)
        test_X=trainMat
        test_y=listClasses
        #GaussianBayes(test_X,test_y)
        MultinomialBayes(trainMat,listClasses,trainMat,listClasses)

if __name__=='__main__':
	#simpleTest()
	spamTest()
