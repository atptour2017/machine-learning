from divide_training import *
from bayesclass import *

attrib_list=['buying','maint','doors','person','lug_boot','safety']

car=Bayes()
#car.initByCompleteFile('car.data')
car.initByTrainFile('r1.data','car.data')
#car.minDivData('car.data',random_div,50)
car.showAllInfo()
car.train()

#data=['low','low','3','4','small','high']	#good
#car.makeDecision(data)
car.makeDecisionForTest()
