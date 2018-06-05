from divide_training import *
from bayesclass import *

xigua=Bayes()
xigua.initByCompleteFile('xigua.data','disperse')
xigua.normalDivData('xigua.data',random_div,50)
xigua.changeContinuousData()
xigua.train()
xigua.showAllInfo()

data=['wuhei','quansuo','chenmen','qingxi','aoxian','yinghua',0.774,0.376]	#good
#xigua.makeDecision(data)
xigua.makeDecisionForTest()
