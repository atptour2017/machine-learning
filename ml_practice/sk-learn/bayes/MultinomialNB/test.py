import numpy as np  
from sklearn.naive_bayes import MultinomialNB  
X = np.array([[1,2,3,4],[1,3,4,4],[2,4,5,5],[2,5,6,5],[3,4,5,6],[3,5,6,6]])  
y = np.array([1,1,4,2,3,3])  
clf = MultinomialNB(alpha=2.0,fit_prior=True,class_prior=[0.3,0.1,0.3,0.2])  
clf.fit(X,y)  
print(clf.class_log_prior_)  
print(np.log(0.3),np.log(0.1),np.log(0.3),np.log(0.2))  


"""
clf1 = MultinomialNB(alpha=2.0,fit_prior=False,class_prior=[0.3,0.1,0.3,0.2])  
clf1.fit(X,y)  
print(clf1.class_log_prior_)  
"""
