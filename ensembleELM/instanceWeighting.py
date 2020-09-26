import numpy as np
import random

def instanceWeighting(datasetX, datasetT, weights):
	# datasetX = np.loadtxt(datasetX, delimiter=',')
	# datasetT = np.loadtxt(datasetT, delimiter=',')
	# D = np.loadtxt(distributionVector, delimiter=',')
	dataset = np.concatenate((datasetX, datasetT), axis=1)
	weights = np.transpose(weights)
	k = datasetX.shape[0]

	data = random.choices(dataset,weights=weights,k=k)
	# data = random.choices(dataset, weights=weights, k=2000)
	npdata = np.vstack(data)
	sample = np.hsplit(npdata, [datasetX.shape[1], dataset.shape[1]])

	sampleX = sample[0]
	sampleT = sample[1]
	return sampleX, sampleT