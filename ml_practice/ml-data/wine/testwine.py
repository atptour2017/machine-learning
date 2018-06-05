from divide_training import *
from bayesclass import *

wine=Bayes()
wine.initByCompleteFile('new_wine.data','disperse')

wine.normalDivData('new_wine.data',random_div,70)
wine.changeContinuousData()
wine.train()
wine.showAllInfo()

"""
data=[6.4,2.8,5.6,2.2]	#good
#iris.makeDecision(data)
"""

wine.makeDecisionForTest()
