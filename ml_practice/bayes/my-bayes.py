from bayes import *
from numpy import *

listOPosts,listClasses=loadDataSet()
print listClasses
myVocablist=createVocabList(listOPosts)
print myVocablist


trainMat=[]
for postinDoc in listOPosts:
        print postinDoc
        tmp=setOfWords2Vec(myVocablist,postinDoc)
        print tmp
        trainMat.append(tmp)

print trainMat

p0V,p1V,pAb=trainNB0(trainMat,listClasses)

print p0V
print p1V
print pAb


