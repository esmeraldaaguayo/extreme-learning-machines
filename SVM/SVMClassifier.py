import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import timeit
import os

def SVMClassifier (file1, file2, file3, file4):
	# load data
	Xtr = np.load(file1)
	Xts = np.load(file2)
	Ttr = np.load(file3)
	Tts = np.load(file4)

	t1 = timeit.default_timer()

	TtsMax = Tts[:,0]


	#  generate model
	model = SVC(kernel='linear', class_weight='balanced')
	# model = LinearSVC(random_state=0, tol=1e-5, class_weight='balanced')
	model.fit(Xtr, Ttr[:, 0])
	predictions = model.predict(Xts)
	# print(predictions)
	YtsMax = predictions
	# print(TtsMax)




	accuracy = (float(np.sum(YtsMax == TtsMax)) / TtsMax.shape[0])
	conf = confusion_matrix(TtsMax, YtsMax)
	tn, fp, fn, tp = confusion_matrix(TtsMax, YtsMax).ravel()
	recall = (float(tp / (tp + fn)))
	precision = (float(tp / (tp + fp)))
	print(accuracy)
	print(recall)
	print(precision)
	print(conf)

	# time
	t2 = timeit.default_timer()
	trainingTime = t2 - t1
	# print(trainingTime)


	return accuracy,recall,precision, trainingTime