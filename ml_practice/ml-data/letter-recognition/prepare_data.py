import csv

file=open('letter-recognition.data','r')
reader=csv.reader(file)
rawlist=list()
for row in reader:
	newlist=list()
	newlist=row[1:]+[row[0]]
	print newlist
	rawlist.append(newlist)
file.close()

file=open('new_letter-recognition.data','w')
writer=csv.writer(file)
#print rawlist

writer.writerows(rawlist)
file.close()

