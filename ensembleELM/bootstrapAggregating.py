import numpy as np
import random
from random import randrange, randint
import pandas as pd

def bootstrapAggregating(datasetX, datasetT, ratio=1.0):
	# datasetX = np.loadtxt(datasetX, delimiter=',')
	# datasetT = np.loadtxt(datasetT, delimiter=',')
	dataset = np.concatenate((datasetX, datasetT), axis=1)

	k = dataset.shape[0]
	data = random.choices(dataset, k=k)
	npdata = np.vstack(data)
	sample = np.hsplit(npdata, [datasetX.shape[1], dataset.shape[1]])

	sampleX = sample[0]
	sampleT = sample[1]
	return sampleX, sampleT