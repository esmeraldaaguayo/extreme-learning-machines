import os
import numpy as np
import pandas as pd
import hpelm
from ensembleELM.bootstrapAggregating import bootstrapAggregating
import timeit

def baggingELM(file1, file2, file3, file4, models):
	# load data
	Xtr = np.load(file1)
	Xts = np.load(file2)
	Ttr = np.load(file3)
	Tts = np.load(file4)

	# keep predictions
	YtsMaxList = []

	basemodels = models

	# train ELMs
	t1 = timeit.default_timer()
	for i in range(basemodels):
		print(i)
		# generate random dataset
		rXtr, rTtr = bootstrapAggregating(Xtr, Ttr)

		model = hpelm.HPELM(rXtr.shape[1], rTtr.shape[1])
		model.add_neurons(1000, 'sigm')

		# prep weight matrix
		w = np.zeros((rTtr.shape[1],))
		w[0] += 9
		w[1] += 1.125

		model.train(rXtr, rTtr, "wc")
		# model.save("ELMmodelWeighted_bagging_%d.h5" % i)

		# make prediction
		# print(Xts[0].reshape(1,-1).shape)
		Yts = model.predict(Xts)

		# evaluate classification results
		YtsMax = np.argmax(Yts, 1)
		listYts =list(YtsMax)
		YtsMaxList.extend([listYts])

	# time
	t2 = timeit.default_timer()
	trainingTime = t2 - t1

	TtsMax = np.argmax(Tts, 1)

	# np.savetxt("baggingELM_predictions_1models.csv", numerator, delimiter=",")
	return YtsMaxList, TtsMax, trainingTime