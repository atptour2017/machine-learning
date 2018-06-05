
import numpy as np  
#X = np.random.randint(2, size=(6, 100))  
#Y = np.array([1, 2, 3, 4, 4, 5])  
X=[[1,1,0,0],[0,0,1,1]]
Y=[0,1]
from sklearn.naive_bayes import BernoulliNB  
clf = BernoulliNB()  
clf.fit(X, Y)  
print clf.predict([[1,1,0,1]])
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)  
#print(clf.predict(X[2:3])) 
