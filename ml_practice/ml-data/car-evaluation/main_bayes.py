from divide_training import *
from bayeslib import *

attrib_list=['buying','maint','doors','person','lug_boot','safety']

datalist,labellist=loadData('car.data')
purelabel=list(set(labellist))
print purelabel

traindat,testdat=begin_div(datalist,99)
trainlabel,testlabel=begin_div(labellist,99)

print mat(traindat)
print mat(trainlabel)

attriblist=createAttribList(datalist)
print attriblist

"""
plabel=calLabelPosibility(trainlabel)
print 'plabel:'
print plabel
print

plist=calTotalPosibility(traindat,trainlabel,attriblist)
print 'plist:'
print plist
print

testdata=['low','low','5more','4','small','low']
print calPosibility(testdata,purelabel,attriblist,plist,plabel)
"""
