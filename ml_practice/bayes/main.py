from bayes import *
from numpy import *

"""
listOPosts,listClasses=loadDataSet()
myVocablist=createVocabList(listOPosts)
print myVocablist
vec0=setOfWords2Vec(myVocablist,listOPosts[0])
vec3=setOfWords2Vec(myVocablist,listOPosts[3])

#print vec0
#print vec3

trainMat=[]
for postinDoc in listOPosts:
	print postinDoc
	tmp=setOfWords2Vec(myVocablist,postinDoc)
	print tmp
	trainMat.append(setOfWords2Vec(myVocablist,postinDoc))

#print trainMat

p0V,p1V,pAb=trainNB0(trainMat,listClasses)

print pAb
print p0V
print p1V
"""

spamTest()
