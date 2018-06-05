from divide_training import *
from decisiontree import *

xigua=DecisionTree('attribute')
xigua.initByCompleteFile('purexigua.data','disperse')

xigua.normalDivData('purexigua.data',random_div,0)
xigua.changeContinuousData()
xigua.showAllInfo()
xigua.C45Train()
xigua.showAllInfo()

data=['wuhei','shaojuan','zhuoxiang','shaohu','shaoao','ruannian',0.481,0.149]
print xigua.C45Predict(data)

#wine.makeDecisionForTest()
