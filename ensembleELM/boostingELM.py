import numpy as np
import hpelm
from ensembleELM.instanceWeighting import instanceWeighting
import timeit

def boostingELM(file1, file2, file3, file4, model):
	# load data
	Xtr = np.load(file1)
	Xts = np.load(file2)
	Ttr = np.load(file3)
	Tts = np.load(file4)

	# initialize weights
	instanceNumber = Xtr.shape[0]
	uniformDist = 1/instanceNumber
	weights = np.ones((1,instanceNumber)) * uniformDist

	# keep
	alphaList =[]
	YtrMaxList = []
	YtsMaxList =[]

	basemodels = model

	t1 = timeit.default_timer()
	for i in range(basemodels):
		# get weighted instances data
		rXtr, rTtr = instanceWeighting(Xtr, Ttr, weights)

		#train model
		model = hpelm.HPELM(rXtr.shape[1], rTtr.shape[1])
		model.add_neurons(1000, 'sigm')

		# prep weight matrix
		w = np.zeros((rTtr.shape[1],))
		w[0] += 9
		w[1] += 1.125

		model.train(rXtr, rTtr, "wc")
		# model.save("ELMmodelWeighted_boosting_%d.h5" % i)

		# make prediction
		Ytr = model.predict(Xtr)
		Yts = model.predict(Xts)

		# evaluate classification results
		YtrMax = np.argmax(Ytr, 1)
		listYtr = list(YtrMax)
		YtrMaxList.extend([listYtr])
		YtsMax = np.argmax(Yts,1)
		listYts = list(YtsMax)
		YtsMaxList.extend([listYts])
		TtrMax = np.argmax(Ttr, 1)

		# majority voted
		join = np.vstack(YtrMaxList)
		sum = np.sum(join, axis=0)
		l = join.shape[0]
		YtrVoted = [1 if l - e < l / 2 else 0 for e in sum]
		YtrVoted = np.array(YtrVoted)

		# update weights
		weights = np.ones((1,instanceNumber))/instanceNumber
		totalError = np.sum(np.multiply((YtrVoted != TtrMax), weights))
		if totalError < 0.5:
			print('pass')
			alpha = (1 / 2) * np.log((1 - totalError) / totalError)
			alphaList.append(alpha)
			seq = [1 if YtrVoted[i] == TtrMax[i] else -1 for i in range(instanceNumber)]
			weights = np.multiply(np.exp(np.multiply(-alpha, seq)), weights)
			weightsum = np.sum(weights)
			weights = weights / weightsum

	TtsMax = np.argmax(Tts, 1)
	t2 = timeit.default_timer()
	trainingTime = t2 - t1

	return YtsMaxList, alphaList, TtsMax, trainingTime