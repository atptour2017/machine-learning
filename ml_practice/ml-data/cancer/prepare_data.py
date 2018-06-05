# -*- coding: UTF-8 -*-

from divide_training import *
import csv
from numpy import *
import copy
import random
import re

def loadCancerData(filename,newfilename):         #从原始数据文件读取数据
        file=open(filename,'r')
        lines=file.readlines()
	rawlist=list()
        for row in lines:
		row=row.strip()
		p=re.compile('[ \t\n]+')
		newrow=re.sub(p,',',row)
		#tmplist=row.strip().split('\t')
		tmplist=newrow.split(',')
		print tmplist
		tmplist=tmplist[2:]+list(tmplist[1])
                rawlist.append(tmplist)
        file.close()
	file=open(newfilename,'w')
	writer=csv.writer(file)
	for row in rawlist:
		writer.writerow(row)
	file.close()

def prepareNewFile(filename,newfilename):
	file=open(filename,'r')
        lines=file.readlines()
        rawlist=list()
        for row in lines:
                row=row.strip()
                p=re.compile('[ \t\n]+')
                newrow=re.sub(p,',',row)
                #tmplist=row.strip().split('\t')
                tmplist=newrow.split(',')
                rawlist.append(tmplist)
        file.close()
        file=open(newfilename,'w')
        writer=csv.writer(file)
        for row in rawlist:
                writer.writerow(row)
        file.close()

#loadCancerData('cancer.data','new_cancer.data')
prepareNewFile('cancer.data','ori_cancer.data')
