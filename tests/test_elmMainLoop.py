import unittest
from Single_ELMs.elmMainLoop import elmMainLoop
import seaborn as sns
import matplotlib.pyplot as plt

class TestElMMainLoop(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.acc, cls.recall, cls.precision = \
			elmMainLoop("../Data/Datasets/CVdatasets/npy_TrainTest_Unstd",
						"../Data/Datasets/CVdatasets/npy_TrainTest_reducedUnstd",
						"../Data/Datasets/CVdatasets/npy_TrainTest_noBaselineUnstd",
						"../Data/Datasets/CVdatasets/npy_TrainTest_Std",
						"../Data/Datasets/CVdatasets/npy_TrainTest_reducedStd",
						"../Data/Datasets/CVdatasets/npy_TrainTest_noBaselineStd")
			# elmMainLoop("../Data/Datasets/globalSplit/unstd/crossValidationDataset",
			# 			"../Data/Datasets/globalSplit/reducedUnstd/crossValidationDataset",
			# 			"../Data/Datasets/globalSplit/noBaselineUnstd/crossValidationDataset",
			# 			"../Data/Datasets/globalSplit/std/crossValidationDataset",
			# 			"../Data/Datasets/globalSplit/reducedStd/crossValidationDataset",
			# 			"../Data/Datasets/globalSplit/noBaselineStd/crossValidationDataset")

	def test_dataframe_formation(self):
		self.assertEqual(self.acc.shape, (20, 6))


	def test_boxplotsAccuracy(self):
		fig, ax1 = plt.subplots(figsize=(9, 4))
		sns.set(style="whitegrid")
		ax1 = sns.boxplot(data=self.acc, color="white")
		ax1 = sns.swarmplot(data=self.acc, color=".25")
		plt.ylabel("Accuracy")
		fig.savefig('fig1Accuracy.png')

	def test_boxplotsRecall(self):
		fig, ax2 = plt.subplots(figsize=(9, 4))
		sns.set(style="ticks")
		ax2 = sns.boxplot(data=self.recall, color="white")
		ax2 = sns.swarmplot(data=self.recall, color=".25")
		plt.ylabel("Recall")
		fig.savefig('fig1Recall.png')

	def test_boxplotsPrecision(self):
		fig, ax3 = plt.subplots(figsize=(9, 4))
		sns.set_style("whitegrid")
		ax3 = sns.boxplot(data=self.precision, color="white")
		ax3 = sns.swarmplot(data=self.precision, color=".25")
		plt.ylabel("Precision")
		fig.savefig('fig1Precision.png')

	def test_boxplotsAccuracyStyle(self):
		fig, ax3 = plt.subplots(figsize=(9, 4))
		sns.set_style("whitegrid")
		ax3 = sns.boxplot(data=self.acc, color="white")
		ax3 = sns.swarmplot(data=self.acc, color=".25")
		plt.ylabel("Accuracy")
		fig.savefig('fig1AccuracyStyle.png')







