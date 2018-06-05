# -*- coding: UTF-8 -*-
from random import randint
import copy

def random_div(raw_list,percent):
	training_list=copy.deepcopy(raw_list)
	length=len(training_list)
	len_test=int(length*percent/100)
	i=0
	test_list=list()
	while i<len_test:
		index=randint(0,len(training_list)-1)
		selected=training_list[index]
		del training_list[index]
		test_list.append(selected)
		i=i+1
	return training_list,test_list

def begin_div(raw_list,percent):	#percent是test的比例，前面是test,后面是training
	training_list=copy.deepcopy(raw_list)
        length=len(training_list)
        len_test=int(length*percent/100)
	test_list=training_list[0:len_test]
	training_list=training_list[len_test:length]
	return training_list,test_list

def end_div(raw_list,percent):		#percent是test的比例，前面是training,后面是test
	training_list=copy.deepcopy(raw_list)
        length=len(training_list)
        len_test=int(length*percent/100)
	print 'len_test=',len_test
        test_list=training_list[length-len_test:length]
        training_list=training_list[0:length-len_test]
        return training_list,test_list

if __name__ == '__main__':
	a=[0,1,2,3,4,5,6,7,8,9]
	print end_div(a,50)

		

