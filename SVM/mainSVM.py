# from majorityVotingSVM import majorityVotingSVM
from SVM.SVMClassifier import SVMClassifier
from boxplotVisualization import boxplotVisualization
from SVM.SVMParameterTuning import SVMParameterTuning
import os
import pandas as pd

folder = os.path.join(os.path.dirname(__file__), "../Data/Datasets/CVdatasets/npy_TrainTest_noBaselineStd")

#keep
votedaccList =[]
votedrecalList =[]
votedprecList = []
trainingTimeList = []

for i in range(1):
	loop = i
	print(loop)

	Xtr = (os.path.join(folder, "noBaselineStd_train_x_%d.npy" % i))
	Xts = (os.path.join(folder, "noBaselineStd_test_x_%d.npy" % i))
	Ttr = (os.path.join(folder, "noBaselineStd_train_t_%d.npy" % i))
	Tts = (os.path.join(folder, "noBaselineStd_test_t_%d.npy" % i))

	# C, trainingTime = SVMParameterTuning(Xtr, Xts, Ttr, Tts)
	# print(trainingTime)
	# print(C)

	acc, recall, prec, trainingTime = SVMClassifier(Xtr, Xts, Ttr, Tts)

	votedaccList.append(acc)
	votedrecalList.append(recall)
	votedprecList.append(prec)
	trainingTimeList.append(trainingTime)

#create dataframes
df_acc = pd.DataFrame(votedaccList, columns=['accuracy'])
df_recal = pd.DataFrame(votedrecalList, columns=['recall'])
df_prec = pd.DataFrame(votedprecList, columns=['precision'])
df_time = pd.DataFrame(trainingTimeList, columns=['training time'])

concatinate_metrics=pd.concat([df_acc,df_recal,df_prec, df_time], axis=1)
metrics = pd.DataFrame(data= concatinate_metrics, columns=['accuracy', 'recall', 'precision', 'training time'])

# save metrics
metrics.to_csv("svm_model2.csv", index=False)

# boxplotVisualization(metrics)

