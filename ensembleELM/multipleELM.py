import numpy as np
import timeit
import hpelm

def multipleELM(file1,file2,file3,file4, models):
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

		model = hpelm.HPELM(Xtr.shape[1], Ttr.shape[1])
		model.add_neurons(1000, 'sigm')

		# prep weight matrix
		w = np.zeros((Ttr.shape[1],))
		w[0] += 9
		w[1] += 1.125

		model.train(Xtr, Ttr, "wc")
		# model.save("ELMmodelWeighted_bagging_%d.h5" % i)

		# make prediction
		Yts = model.predict(Xts)

		# evaluate classification results
		YtsMax = np.argmax(Yts, 1)
		listYts = list(YtsMax)
		YtsMaxList.extend([listYts])

	# time
	t2 = timeit.default_timer()
	trainingTime = t2 - t1

	TtsMax = np.argmax(Tts, 1)

	return YtsMaxList, TtsMax, trainingTime