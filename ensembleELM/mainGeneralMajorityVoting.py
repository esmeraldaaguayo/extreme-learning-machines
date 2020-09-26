from ensembleELM.modelPredictions import modelPredictions
from ensembleELM.generalMajorityVoting import generalMajorityVoting
from boxplotVisualization import boxplotVisualization

YtsList, actual = modelPredictions("../Data/Datasets/globalSplit/noBaselineStd/Xts_noBaselineStd.npy",
								   "../Data/Datasets/globalSplit/noBaselineStd/Tts_noBaselineStd.npy")
print(len(YtsList))
print(YtsList[0])

metrics, accuracyVoted, recallVoted, presicionVoted, trainingTime = generalMajorityVoting(YtsList, actual)

print(accuracyVoted)
print(recallVoted)
print(presicionVoted)
print(trainingTime)

boxplotVisualization(metrics)