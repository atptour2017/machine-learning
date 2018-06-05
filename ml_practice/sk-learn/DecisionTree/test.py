from sklearn import tree
#X = [[0, 0], [1, 1]]
X=[['a','a'],['b','b']]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)
print clf.predict([['a', 'a']])
