import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import timeit

def generalMajorityVoting(YtsList,TtsArray, m):

	m = m
	accuracyDict = []
	recallDict = []
	precisionDict = []

	# individuals
	for i in range(m):
		TtsMax = TtsArray
		YtsMax = YtsList[i]
		accuracyDict.append((float(np.sum(YtsMax == TtsMax)) / TtsMax.shape[0]))
		tp, fn, fp, tn = confusion_matrix(TtsMax, YtsMax).ravel()
		recallDict.append((float(tp / (tp + fn))))
		precisionDict.append(float(tp / (tp + fp)))

	# majority voted
	join = np.vstack(YtsList)
	sum = np.sum(join, axis=0)
	len = join.shape[0]
	YtrVoted = [1 if len - elem < len / 2 else 0 for elem in sum]
	# print(YtrVoted)
	YtrVoted = np.array(YtrVoted)
	TtsMax = TtsArray

	# evaluate metrics
	accuracyVoted = (float(np.sum(YtrVoted == TtsMax)) / TtsMax.shape[0])
	tp, fn, fp, tn = confusion_matrix(TtsMax, YtrVoted).ravel()
	recallVoted = (float(tp / (tp + fn)))
	presicionVoted = (float(tp / (tp + fp)))

	accMetrics = pd.DataFrame(accuracyDict, columns=['accuracy'])
	recalMetrics = pd.DataFrame(recallDict, columns=['recall'])
	precMetrics = pd.DataFrame(precisionDict, columns=['precision'])

	concatinate_metrics = pd.concat([accMetrics, recalMetrics, precMetrics], axis=1)
	metrics = pd.DataFrame(data=concatinate_metrics, columns=['accuracy', 'recall', 'precision'])

	# save metrics
	metrics.to_csv("majorityELM_individual_metrics_{}_model.csv".format(m), index=False)

	return metrics, accuracyVoted, recallVoted, presicionVoted