import pandas as pd
import numpy as np
import random

def globalSplit(csv_file_x, csv_file_t):
	# load data
	X = np.loadtxt(csv_file_x, delimiter=',')
	T = np.loadtxt(csv_file_t, delimiter=',')
	print('pass')

	# dataset arranged in subject arrays
	indices = [np.arange(0, 4860), np.arange(4860, 9720), np.arange(9720, 14580), np.arange(14580, 19440),
			   np.arange(19440, 24165), np.arange(24165, 29025), np.arange(29025, 33885), np.arange(33885, 38745),
			   np.arange(38745, 43605), np.arange(43605, 48465), np.arange(48465, 53325), np.arange(53325, 58185),
			   np.arange(58185, 63045), np.arange(63045, 67905), np.arange(67905, 72765), np.arange(72765, 77490),
			   np.arange(77490, 82350), np.arange(82350, 87210), np.arange(87210, 92070), np.arange(92070, 96930)]

	# randomly select two subject
	testList =[12,1]
	# for i in range(2):
	# 	testList.append(random.randint(0,19))
	print(testList)

	# partition data based on indices
	items = [j for j in range(20) if j not in testList]
	print(items)

	trainIndeces = np.hstack([indices[j] for j in items])
	testIndeces = np.hstack([indices[j] for j in testList])
	print(trainIndeces.shape)
	print(testIndeces.shape)

	Xtr = X[trainIndeces]
	Ttr = T[trainIndeces]
	Xts = X[testIndeces]
	Tts = T[testIndeces]
	# print(Xtr)
	# print(Ttr)
	# print(Xts)
	# print(Tts)
	# Xtr = X[items]
	# Xts = X[testList]
	# Ttr = T[items]
	# Tts = T[testList]

	# save files
	# np.savetxt("Xtr_noBaselineStd.csv", Xtr, delimiter=",")
	# np.savetxt("Xts_noBaselineStd.csv", Xts, delimiter=",")
	# np.savetxt("Ttr_noBaselineStd.csv", Ttr, delimiter=",")
	# np.savetxt("Tts_noBaselineStd.csv", Tts, delimiter=",")
	# np.save("Xtr_noBaselineStd.npy", Xtr)
	# np.save("Xts_noBaselineStd.npy", Xts)
	# np.save("Ttr_noBaselineStd.npy", Ttr)
	# np.save("Tts_noBaselineStd.npy", Tts)

	# np.savetxt("Xtr_noBaselineUnstd.csv", Xtr, delimiter=",")
	# np.savetxt("Xts_noBaselineUnstd.csv", Xts, delimiter=",")
	# np.savetxt("Ttr_noBaselineUnstd.csv", Ttr, delimiter=",")
	# np.savetxt("Tts_noBaselineUnstd.csv", Tts, delimiter=",")
	# np.save("Xtr_noBaselineUnstd.npy", Xtr)
	# np.save("Xts_noBaselineUnstd.npy", Xts)
	# np.save("Ttr_noBaselineUnstd.npy", Ttr)
	# np.save("Tts_noBaselineUnstd.npy", Tts)

	# np.savetxt("Xtr_reducedUnstd.csv", Xtr, delimiter=",")
	# np.savetxt("Xts_reducedUnstd.csv", Xts, delimiter=",")
	# np.savetxt("Ttr_reducedUnstd.csv", Ttr, delimiter=",")
	# np.savetxt("Tts_reducedUnstd.csv", Tts, delimiter=",")
	# np.save("Xtr_reducedUnstd.npy", Xtr)
	# np.save("Xts_reducedUnstd.npy", Xts)
	# np.save("Ttr_reducedUnstd.npy", Ttr)
	# np.save("Tts_reducedUnstd.npy", Tts)

	# np.savetxt("Xtr_reducedStd.csv", Xtr, delimiter=",")
	# np.savetxt("Xts_reducedStd.csv", Xts, delimiter=",")
	# np.savetxt("Ttr_reducedStd.csv", Ttr, delimiter=",")
	# np.savetxt("Tts_reducedStd.csv", Tts, delimiter=",")
	# np.save("Xtr_reducedStd.npy", Xtr)
	# np.save("Xts_reducedStd.npy", Xts)
	# np.save("Ttr_reducedStd.npy", Ttr)
	# np.save("Tts_reducedStd.npy", Tts)

	# np.savetxt("Xtr_Std.csv", Xtr, delimiter=",")
	# np.savetxt("Xts_Std.csv", Xts, delimiter=",")
	# np.savetxt("Ttr_Std.csv", Ttr, delimiter=",")
	# np.savetxt("Tts_Std.csv", Tts, delimiter=",")
	# np.save("Xtr_Std.npy", Xtr)
	# np.save("Xts_Std.npy", Xts)
	# np.save("Ttr_Std.npy", Ttr)
	# np.save("Tts_Std.npy", Tts)

	np.savetxt("Xtr_Unstd.csv", Xtr, delimiter=",")
	np.savetxt("Xts_Unstd.csv", Xts, delimiter=",")
	np.savetxt("Ttr_Unstd.csv", Ttr, delimiter=",")
	np.savetxt("Tts_Unstd.csv", Tts, delimiter=",")
	np.save("Xtr_Unstd.npy", Xtr)
	np.save("Xts_Unstd.npy", Xts)
	np.save("Ttr_Unstd.npy", Ttr)
	np.save("Tts_Unstd.npy", Tts)

	return X, T, Xtr, Xts, Ttr, Tts
