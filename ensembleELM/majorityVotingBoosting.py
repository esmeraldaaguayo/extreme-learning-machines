import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def majorityVotingBoosting(YtsList, alphaList, TtsArray):
	# YtsList =YtsList
	# alphaList=alphaList
	# YtsList = [np.array([0,0,0,1]), np.array([1,1,1,1])]
	# alphaList = [0.54, 0.23]
	# TtsArray = np.array([1,1,1,1])
	TtsMax = TtsArray

	# individual
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
	weightedVoteList = list(range(TtsArray.shape[0]))
	weightedVote0 = 0
	weightedVote1 = 0
	for j in range(join.shape[1]):
		for i in range(join.shape[0]):
			if join[i][j] == 0:
				weightedVote0 += alphaList[i]
			if join[i][j] == 1:
				weightedVote1 += alphaList[i]
		votes =[weightedVote0, weightedVote1] #[0,1] classes
		weightedVote = np.argmax(votes)
		weightedVoteList[j] = weightedVote
		weightedVote0 = 0
		weightedVote1 = 0

	YtsVoted = np.array(weightedVoteList)
	# print(len(weightedVoteList))
	# print(weightedVoteList)
	# print(weightedVote0)
	# print(weightedVote1)

	# evaluate metrics
	accuracyVoted = (float(np.sum(YtsVoted == TtsMax)) / TtsMax.shape[0])
	tp, fn, fp, tn = confusion_matrix(TtsMax, YtsVoted).ravel()
	recallVoted = (float(tp / (tp + fn)))
	presicionVoted = (float(tp / (tp + fp)))

	accMetrics = pd.DataFrame(accuracyDict, columns=['accuracy'])
	recalMetrics = pd.DataFrame(recallDict, columns=['recall'])
	precMetrics = pd.DataFrame(precisionDict, columns=['precision'])

	concatinate_metrics = pd.concat([accMetrics, recalMetrics, precMetrics], axis=1)
	metrics = pd.DataFrame(data=concatinate_metrics, columns=['accuracy', 'recall', 'precision'])

	# save metrics
	metrics.to_csv("boostingELM_individual_metrics_{}_model.csv".format(m), index=False)

	return metrics, accuracyVoted, recallVoted, presicionVoted