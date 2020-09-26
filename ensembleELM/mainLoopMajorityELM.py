import os
import pandas as pd
from ensembleELM.generalMajorityVoting import generalMajorityVoting
from ensembleELM.multipleELM import multipleELM
from boxplotVisualization import boxplotVisualization

folder = os.path.join(os.path.dirname(__file__), "../Data/Datasets/CVdatasets/npy_TrainTest_noBaselineStd")
models = 100

#keep
votedaccList =[]
votedrecalList =[]
votedprecList = []
trainingTimeList = []

for i in range(20):
	loop = i
	print(loop)

	Xtr = (os.path.join(folder, "noBaselineStd_train_x_%d.npy" % i))
	Xts = (os.path.join(folder, "noBaselineStd_test_x_%d.npy" % i))
	Ttr = (os.path.join(folder, "noBaselineStd_train_t_%d.npy" % i))
	Tts = (os.path.join(folder, "noBaselineStd_test_t_%d.npy" % i))

	YtsList, TtsArray, trainingTime = multipleELM(Xtr, Xts, Ttr, Tts, models)

	trainingTimeList.append(trainingTime)
	m = len(YtsList)

	metrics, votedacc, votedrecal, votedprec = generalMajorityVoting(YtsList,TtsArray, m)

	votedaccList.append(votedacc)
	votedrecalList.append(votedrecal)
	votedprecList.append(votedprec)

#create dataframes
df_acc = pd.DataFrame(votedaccList, columns=['accuracy'])
df_recal = pd.DataFrame(votedrecalList, columns=['recall'])
df_prec = pd.DataFrame(votedprecList, columns=['precision'])
df_time = pd.DataFrame(trainingTimeList, columns=['training time'])

concatinate_metrics=pd.concat([df_acc,df_recal,df_prec, df_time], axis=1)
votedmetrics = pd.DataFrame(data= concatinate_metrics, columns=['accuracy', 'recall', 'precision', 'training time'])

# save metrics
votedmetrics.to_csv("majorityELM_voted_metrics_{}_model.csv".format(models), index=False)

boxplotVisualization(votedmetrics)
