from divide_training import *
from bayesclass import *

cancer=Bayes()
cancer.initByCompleteFile('new_cancer.data','disperse')

cancer.normalDivData('new_cancer.data',random_div,50)
cancer.changeContinuousData()
cancer.train()
cancer.showAllInfo()

"""
data=[6.4,2.8,5.6,2.2]	#good
#iris.makeDecision(data)
"""

cancer.makeDecisionForTest()
