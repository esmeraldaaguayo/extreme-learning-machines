import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def majorityVotingBagging (YtsList, TtsArray,loop):
	# YtsList = YtsList
	# TtsArray = TtsArray
	# YtsList = [np.array([1,1,1]), np.array([0,1,0])]
	# TtsArray = np.array([1,1,0])

	loop=loop
	m = len(YtsList)
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
	l = join.shape[0]
	YtrVoted = [1 if l - e < l / 2 else 0 for e in sum]
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

	concatinate_metrics=pd.concat([accMetrics, recalMetrics, precMetrics], axis=1)
	metrics = pd.DataFrame(data= concatinate_metrics, columns=['accuracy', 'recall', 'precision'])

	# save metrics
	metrics.to_csv("baggingELM_individual_metrics_{}_iteration.csv".format(loop), index=False)

	return metrics, accuracyVoted, recallVoted, presicionVoted