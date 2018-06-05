# -*- coding: UTF-8 -*-

import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)
data=[[-0.8, -1]]
print(clf.predict(data))
print clf.class_prior_		#先验概率
print clf.class_count_		#样本数
print clf.theta_		#样本均值
print clf.sigma_ 		#方差
#clf.set_params(priors=[ 0.625,0.375])  #设置先验参数
print clf.get_params()
print clf.score(X,Y)

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
#print(clf_pf.predict(data))
