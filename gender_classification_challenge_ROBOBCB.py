import numpy as np
from sklearn.metrics import accuracy_score #precision_score
from sklearn import tree, neighbors, svm, naive_bayes
import textwrap

clf = tree.DecisionTreeClassifier()
# CHALLENGE - create 3 more classifiers...
clf1 = neighbors.KNeighborsClassifier()
clf2 = svm.SVC()
clf3 = naive_bayes.GaussianNB()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# CHALLENGE - ...and train them on our data
clf_trained = clf.fit(X, Y)
clf1_trained = clf1.fit(X, Y)
clf2_trained = clf2.fit(X, Y)
clf3_trained = clf3.fit(X, Y)

new_data = np.array([170, 70, 43])
prediction_array = np.array([clf_trained.predict([new_data]), clf1_trained.predict([new_data]), clf2_trained.predict([new_data]), clf3_trained.predict([new_data])])

# CHALLENGE compare their reusults and print the best one!
prediction_score = [clf_trained.score(X, Y), clf1_trained.score(X, Y), clf2_trained.score(X, Y), clf3_trained.score(X, Y)]
index = prediction_score.index(max(prediction_score))
prediction_best = prediction_array[index]

#Print out name and perdiction of best method
names = [str(clf), str(clf1), str(clf2), str(clf3)]
#This print line gives the best guess
print ("Given %i cm height, %i kgs weight, and a %i shoe size; the ['%s'] AI predicted the data was from a %s" %(new_data[0], new_data[1], new_data[2], textwrap.shorten(names[index], width=50), prediction_best))

#This print line gives all guesses
#for x in range (4):
#	print ("score: %i, name: %s, prediction: %s" %(prediction_score[x], textwrap.shorten(names[x], width=50), prediction_array[x]))
