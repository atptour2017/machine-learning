from divide_training import *
from decisiontree import *

car=DecisionTree('attribute')
car.initByCompleteFile('car.data','disperse')

car.normalDivData('car.data',random_div,50)

#car.changeContinuousData()
#car.showAllInfo()
car.ID3Train()
#car.C45Train()

car.showAllInfo()
car.ID3PredictTest()
"""

car.storeID3Tree('id3tree')

car1=DecisionTree('attribute')
car1.initByCompleteFile('car.data','disperse')
car1.restoreID3Tree('id3tree')
print car1.ID3Tree

#data=['med','med','2','4','big','high']
#print car1.ID3Predict(data)

rawlist=loadData('car.data')
car1.testdat=rawlist

car1.ID3PredictTest()
"""
