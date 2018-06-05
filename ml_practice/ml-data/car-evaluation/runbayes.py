from divide_training import *
from bayesclass import *

attrib_list=['buying','maint','doors','person','lug_boot','safety']

car=Bayes()
car.initByTrainFile('r1.data')
car.showTrainData()

data=['vhigh','low','2','4','big','med']
#car.makeDecision(data)

#car.makeDecisionForTest()

#car.generateRandomData(95,'r1.data')

"""
testdata=['low','low','5more','4','small','low']
car.makeDecision(testdata)
"""
"""
testdata=[['low','low','5more','4','small','low'],
        ['low','low','5more','4','small','med'],
        ['low','low','5more', '4', 'small', 'high'],
        ['low', 'low', '5more', '4', 'med', 'low'],
        ['low', 'low', '5more', '4', 'med', 'med'],
        ['low', 'low', '5more', '4', 'med', 'high'],
        ['low', 'low', '5more', '4', 'big', 'low'],
        ['low', 'low', '5more', '4', 'big', 'med'],
        ['low', 'low', '5more', '4', 'big', 'high'],
        ['low', 'low', '5more', 'more', 'small', 'low'],
        ['low', 'low', '5more', 'more', 'small', 'med'],
        ['low', 'low', '5more', 'more', 'small', 'high'],
        ['low', 'low', '5more', 'more', 'med', 'low'],
        ['low', 'low', '5more', 'more', 'med', 'med'],
        ['low', 'low', '5more', 'more', 'med', 'high'],
        ['low', 'low', '5more', 'more', 'big', 'low'],
        ['low', 'low', '5more', 'more', 'big', 'med'],
        ['low', 'low', '5more', 'more', 'big', 'high']]
for data in testdata:
        car.makeDecision(data)
"""
