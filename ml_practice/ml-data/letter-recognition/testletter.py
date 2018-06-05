from divide_training import *
from bayesclass import *

letter=Bayes()
letter.initByCompleteFile('new_letter-recognition.data','disperse')
#letter.normalDivData('new_letter-recognition.data',random_div,60)
letter.minDivData('new_letter-recognition.data',random_div,50)
letter.changeContinuousData()
letter.train()
letter.showAllInfo()

data=['wuhei','quansuo','chenmen','qingxi','aoxian','yinghua',0.774,0.376]	#good
#xigua.makeDecision(data)
letter.makeDecisionForTest()
