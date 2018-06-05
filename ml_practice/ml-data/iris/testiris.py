from divide_training import *
from bayesclass import *

iris=Bayes()
iris.initByCompleteFile('iris.data','disperse')
iris.normalDivData('iris.data',random_div,50)
iris.changeContinuousData()
iris.train()
iris.showAllInfo()

data=[6.4,2.8,5.6,2.2]	#good
#iris.makeDecision(data)
iris.makeDecisionForTest()
