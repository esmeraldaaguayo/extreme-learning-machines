import os
import numpy as np
import hpelm
import timeit
from sklearn.metrics import confusion_matrix
import pandas as pd
import csv


def auxELMStdNoBaseline(folder1, hiddenNeurons, neuronType):
	# def auxELMStdNoBaseline(folder1, folder2, hiddenNeurons, neuronType):
	# Xts = np.load(os.path.join(folder2, "Xts_noBaselineStd.npy"))
	# Tts = np.load(os.path.join(folder2, "Tts_noBaselineStd.npy"))
	folder1 = os.path.join(os.path.dirname(__file__), folder1)
	# folder2 = os.path.join(os.path.dirname(__file__), folder2)
	accuracy = []
	recall = []
	precision = []
	trainingTimeList = []
	YtsMaxList = []
	TtsMaxList = []
	confList = []

	for i in range(20):
		Xtr = np.load(os.path.join(folder1, "noBaselineStd_train_x_%d.npy" % i))
		Xts = np.load(os.path.join(folder1, "noBaselineStd_test_x_%d.npy" % i))
		Ttr = np.load(os.path.join(folder1, "noBaselineStd_train_t_%d.npy" % i))
		Tts = np.load(os.path.join(folder1, "noBaselineStd_test_t_%d.npy" % i))

		# prep weight matrix
		w = np.zeros((Ttr.shape[1],))
		w[0] += 9
		w[1] += 1.125

		# train model
		model = hpelm.HPELM(Xtr.shape[1], Ttr.shape[1])
		model.add_neurons(hiddenNeurons, neuronType)

		# training speed
		t1 = timeit.default_timer()

		model.train(Xtr, Ttr, 'wc', w=w)
		t2 = timeit.default_timer()
		trainingTime = t2 - t1
		trainingTimeList.append(trainingTime)

		model.save("ELMmodelWeighted_1000N_noBaselineStd_%d.h5" % i)

		# make prediction
		Yts = model.predict(Xts)

		# evaluate classification results
		TtsMax = np.argmax(Tts, 1)
		listTts = list(TtsMax)
		TtsMaxList.append([listTts])
		YtsMax = np.argmax(Yts, 1)
		listYts = list(YtsMax)
		YtsMaxList.append([listYts])

		accuracy.append(float(np.sum(YtsMax == TtsMax)) / TtsMax.shape[0])
		tp, fn, fp, tn = confusion_matrix(TtsMax, YtsMax).ravel()
		recall.append(float(tp / (tp + fn)))
		precision.append(float(tp / (tp + fp)))
		conf = confusion_matrix(TtsMax, YtsMax)
		confList.append([conf])

	# save metrics vectors
	df_accuracy = pd.DataFrame(data=accuracy, columns=["accuracy"])
	df_recall = pd.DataFrame(data=recall, columns=["recall"])
	df_precision = pd.DataFrame(data=precision, columns=["precision"])
	df_time = pd.DataFrame(data=trainingTimeList, columns=["training time"])
	concatinate_metrics = pd.concat([df_accuracy, df_recall, df_precision, df_time], axis=1)
	df_metrics = pd.DataFrame(data=concatinate_metrics, columns=["accuracy", "recall", "precision", "training time"])

	df_metrics.to_csv("noBaselineStd_CV_metrics.csv", index=False)

	with open('predictionsNoBaselineStdSingleELM.txt', 'w') as p:
		csvwriter1 = csv.writer(p)
		csvwriter1.writerows(YtsMaxList)

	with open('targetsNoBaselineStdSingleELM.txt', 'w') as k:
		csvwriter2 = csv.writer(k)
		csvwriter2.writerows(TtsMaxList)

	with open('confusionMatricesNoBaselineStdSingleELM.txt', 'w') as f:
		csvwriter3 = csv.writer(f)
		csvwriter3.writerows(confList)

	return accuracy, recall, precision