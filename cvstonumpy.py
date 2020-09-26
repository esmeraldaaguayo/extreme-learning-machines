import numpy as np
import math
import os
import pandas as pd
# folder = os.path.join(os.path.dirname( __file__), "Data/Datasets/npy_TrainTest_Std")
# TTtr = np.load(os.path.join(folder, "test_x_1.npy"))
# print(TTtr.shape)
# Ttr = np.loadtxt("Data/Datasets/globalSplit/Ttr_std.csv", delimiter=',') #1000,2
# Xtr = np.loadtxt("Data/Datasets/globalSplit/Xtr_std.csv", delimiter=',') #1000,2
# print(Ttr)
# T=Ttr
# X=Xtr
# l=T[0:3].sum(axis=0)
# print(l.shape)
# ns = np.zeros((2,))
# for b in range(88):  # batch sum is much faster
# 	start = b * 1000 + 0
# 	print(start)
	# stop = min((b + 1) * 1000, 87237)
	# print(stop)
	# ns += T[start:stop].sum(axis=0) # creates(2,) array
# print(ns) # [ 98. 902.]
# wc = ns.sum() / ns
# print(ns.sum()) # 1000.0
# print(wc) # [10.20408163  1.10864745]

# classification = 'wc'
# wc_vector = None
# for b in range(88):
# 	start = b*1000 #0
# 	stop = min((b+1)*1000, 87237) #1000
# 	Xb = X[start:stop]
# 	Tb = T[start:stop]
# 	break
	# if classification == "wc":# print(Xb.shape)
	# 	print(wc)
	# 	print([np.where(Tb==1)])
	# 	print([np.where(Tb == 1)[1]])
	# 	print(wc[np.where(Tb == 1)[1]])
	# 	wc_vector = wc[np.where(Tb == 1)[1]]
	# print(Xb.shape)
	# print(Tb.shape)
	# print(wc_vector.shape)
	# w = np.array(wc_vector ** 0.5)[:, None]
	# print(Xb.shape)
	# print(Tb.shape)
	# print(wc_vector.shape)
	# print(w.shape)
	# print(wc_vector ** 0.5)
	# print(np.array(wc_vector ** 0.5))
	# print(np.array(wc_vector ** 0.5)[:,None])
	# break

YtrMax = np.loadtxt("tests/Data/YtrMax.csv", delimiter=',')
TtrMax = np.loadtxt("tests/Data/TtrMax.csv", delimiter=',')
Dvector = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
print(YtrMax)
print(TtrMax)

misclass = (YtrMax != TtrMax) #np.sum((YtrMax != TtrMax)*Dvector)#/Dvector
# print(misclass)
weightedmisclass = (YtrMax != TtrMax) * Dvector
# print(weightedmisclass)
totalError = np.sum((YtrMax != TtrMax)*Dvector)/np.sum(Dvector)
# print(totalError)
if totalError < 0.5:
	a = (1 / 2) * math.log((1 - totalError) / totalError)
	# print(a)
	# exp = math.exp(-a * YtrMax[0] * TtrMax[0])
	# print(exp)
	x0 = math.exp(-a * YtrMax[0] * TtrMax[0]) * Dvector[0]
	print(x0)
	x1 = math.exp(-a * YtrMax[1] * TtrMax[1]) * Dvector[1]
	print(x1)
	x2 = math.exp(-a * YtrMax[2] * TtrMax[2])*Dvector[2]
	print(x2)
	x3 = math.exp(-a * YtrMax[3] * TtrMax[3]) * Dvector[3]
	print(x3)
	x4 = math.exp(-a * YtrMax[4] * TtrMax[4]) * Dvector[4]
	print(x4)
	sum = x1+x2+x3+x4
	print(sum)

	x0 = math.exp(-a * YtrMax[0] * -1) * Dvector[0]
	print(x0)
	x1 = math.exp(-a * YtrMax[1] * -1) * Dvector[1]
	print(x1)
	x2 = math.exp(-a * YtrMax[2] * TtrMax[2]) * Dvector[2]
	print(x2)
	x3 = math.exp(-a * YtrMax[3] * TtrMax[3]) * Dvector[3]
	print(x3)
	x4 = math.exp(-a * YtrMax[4] * TtrMax[4]) * Dvector[4]
	print(x4)
	sum = x1 + x2 + x3 + x4
	print(sum)

# exp3 = math.exp(-a * -1 * 1)
# print(exp3)
#
# exp4 = math.exp(-a * 1 * 1)
# print(exp4)
	# top = (-a * YtrMax * TtrMax)
	# print(top)
	# new = math.exp(-a * YtrMax * TtrMax)
# 	for i in range(Dvector.shape[0]):
# 		Dvector[i] = Dvector[i] * math.exp(-a * YtrMax[i] * TtrMax[i])

# print(Dvector)
# m =Dvector/np.sum(Dvector)
# print(np.sum(m))
# print(np.sum(Dvector) )


# flip
# Xtr = np.loadtxt("tests/Data/data.csv", delimiter=',') #1000,2
# print(Xtr.shape[0])
# D = 1/Xtr.shape[0]
# D = 1/87237
# print(D)
# Dvector = np.ones((30,1))* D
# print(Dvector.shape)
