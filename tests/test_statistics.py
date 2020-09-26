import unittest
from metricsStats import metricsStats
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation

class TestmetricsStats(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.acc, cls.recall, cls.precision, cls.accSigttest, cls.recallSigttest, cls.precisionSigttest = \
			metricsStats("../Single_ELMs/outCV/metrics/noBaselineStd_CV_metrics.csv",
						 "../ensembleELM/outCV/generalMajorityVoting/model100/majorityELM_voted_metrics_100_model.csv",
						 "../ensembleELM/outCV/baggingELM/model100/baggingELM_voted_metrics_100_model.csv",
						 "../ensembleELM/outCV/boostingELM/model100/boostingELM_voted_metrics_100_model.csv",
						 "../SVM/svm_model.csv")
			# metricsStats("../ensembleELM/outCV/generalMajorityVoting/model10/majorityELM_voted_metrics_10_model.csv",
			# 			 "../ensembleELM/outCV/generalMajorityVoting/model50/majorityELM_voted_metrics_50_model.csv",
			# 			 "../ensembleELM/outCV/generalMajorityVoting/model100/majorityELM_voted_metrics_100_model.csv")
			# metricsStats("../ensembleELM/outCV/boostingELM/model10/boostingELM_voted_metrics_10_model.csv",
			# 			 "../ensembleELM/outCV/boostingELM/model50/boostingELM_voted_metrics_50_model.csv",
			# 			 "../ensembleELM/outCV/boostingELM/model100/boostingELM_voted_metrics_100_model.csv")
		# metricsStats("../ensembleELM/outCV/baggingELM/model10/baggingELM_voted_metrics_10_model.csv",
		# 			 "../ensembleELM/outCV/baggingELM/model50/baggingELM_voted_metrics_50_model.csv",
		# 			 "../ensembleELM/outCV/baggingELM/model100/baggingELM_voted_metrics_100_model.csv")
		# metricsStats("../Single_ELMs/outCV/metrics/Unstd_CV_metrics.csv",
		# 			 "../Single_ELMs/outCV/metrics/reducedUnstd_CV_metrics.csv",
		# 			 "../Single_ELMs/outCV/metrics/noBaselineUnstd_CV_metrics.csv",
		# 			 "../Single_ELMs/outCV/metrics/Std_CV_metrics.csv",
		# 			 "../Single_ELMs/outCV/metrics/reducedStd_CV_metrics.csv",
		# 			 "../Single_ELMs/outCV/metrics/noBaselineStd_CV_metrics.csv")

			# metricsStats("../ensembleELM/out2subjtest/boostingELM/boostingELM_metrics_10_model.csv",
			# 			 "../ensembleELM/out2subjtest/boostingELM/boostingELM_metrics_50_model.csv",
			# 			 "../ensembleELM/out2subjtest/boostingELM/boostingELM_metrics_100_model.csv",
			# 			 "../ensembleELM/out2subjtest/boostingELM/boostingELM_metrics_500_model.csv")

		# metricsStats("../ensembleELM/out2subjtest/baggingELM/baggingELM_metrics_10_model.csv",
		# 			 "../ensembleELM/out2subjtest/baggingELM/baggingELM_metrics_50_model.csv",
		# 			 "../ensembleELM/out2subjtest/baggingELM/baggingELM_metrics_100_model.csv",
		# 			 "../ensembleELM/out2subjtest/baggingELM/baggingELM_metrics_500_model.csv")

			# metricsStats("../Single_ELMs/out2subjtest/metrics/Unstd_metrics.csv",
			# 			 "../Single_ELMs/out2subjtest/metrics/reducedUnstd_metrics.csv",
			# 			 "../Single_ELMs/out2subjtest/metrics/noBaselineUnstd_metrics.csv",
			# 			 "../Single_ELMs/out2subjtest/metrics/Std_metrics.csv",
			# 			 "../Single_ELMs/out2subjtest/metrics/reducedStd_metrics.csv",
			# 			 "../Single_ELMs/out2subjtest/metrics/noBaselineStd_metrics.csv")



	# box_pairs = [("10 models", "50 models"), ("10 models", "100 models"),
	# 			 ("10 models", "500 models")]
	# box_pairs = [("10 models", "50 models"), ("10 models", "100 models"),
	# 			 ("50 models", "100 models")]
	# box_pairs = [("no Baseline Std", "Unstd"), ("no Baseline Std", "Reduced Unstd"),
	# 			 ("no Baseline Std", "no Baseline Unstd"), ("no Baseline Std", "Std")],
	# box_pairs = [("single ELMs", "majority voting ELMs"), ("single ELMs", "bagging-based ELMs"),
	# 			 ("single ELMs", "boosting-based ELMs"), ("majority voting ELMs", "boosting-based ELMs"),
	# 			 ("bagging-based ELMs", "boosting-based ELMs")],

	def test_dataframe_formation(self):
		pass
		# self.assertEqual(self.acc.shape, (20,6))
		# self.assertEqual(self.recall.shape, (20, 6))
		# self.assertEqual(self.precision.shape, (20, 6))

	def test_ttest_accuracy(self):
		print(self.accSigttest.shape)
		print(self.recallSigttest.shape)
		print(self.precisionSigttest.shape)

	def test_sig_boxplotAccuracy(self):
		# print(self.acc.columns)
		fig, ax3 = plt.subplots(figsize=(9, 7))
		sns.set_style("whitegrid")
		ax3 = sns.boxplot(data=self.acc, color="white")
		ax3 = sns.swarmplot(data=self.acc, color=".25")
		# plt.ylabel("single ELMs Accuracy")
		# plt.ylabel("Boosting ELM Accuracy")
		# plt.ylabel("Bagging ELM Accuracy")
		# plt.ylabel("Majority ELM Accuracy")
		plt.ylabel("Accuracy Comparison")

		# statistical notation
		add_stat_annotation(ax3, data=self.acc,
							box_pairs=[("single ELMs", "majority voting ELMs"), ("single ELMs", "bagging-based ELMs"),
									   ("single ELMs", "boosting-based ELMs"), ("majority voting ELMs", "boosting-based ELMs"),
									   ("bagging-based ELMs", "boosting-based ELMs"), ("boosting-based ELMs", "SVMs")],
							test='t-test_ind', text_format='star', loc='inside', verbose=2)

		# plt.show(ax3)
		# fig.savefig('fig2SingleELMAccuracyStats.png')
		# fig.savefig('fig3BaggingELMAccuracyStats.png')
		# fig.savefig('fig4BoostingELMAccuracyStats.png')
		# fig.savefig('fig5MajorityELMAccuracyStats.png')
		fig.savefig('fig6comparisonELMSVMAccuracyStats.png')

	def test_sig_boxplotPrecision(self):
		# print(self.acc.columns)
		fig4, ax4 = plt.subplots(figsize=(9, 7))
		sns.set_style("whitegrid")
		ax4 = sns.boxplot(data=self.precision, color="white")
		ax4 = sns.swarmplot(data=self.precision, color=".25")
		# plt.ylabel("single ELMs Precision")
		# plt.ylabel("Bagging ELM Precision")
		# plt.ylabel("Boosting ELM Precision")
		# plt.ylabel("Majority ELM Precision")
		plt.ylabel("Precision Comparison")

		# statistical notation
		add_stat_annotation(ax4, data=self.precision,
							box_pairs=[("single ELMs", "majority voting ELMs"), ("single ELMs", "bagging-based ELMs"),
									   ("single ELMs", "boosting-based ELMs"), ("majority voting ELMs", "boosting-based ELMs"),
									   ("bagging-based ELMs", "boosting-based ELMs"), ("boosting-based ELMs", "SVMs")],
							test='t-test_ind', text_format='star', loc='inside', verbose=2)

		# plt.show(ax4)
		# fig4.savefig('fig2SingleELMPrecisionStats.png')
		# fig4.savefig('fig3BaggingELMPrecisionStats.png')
		# fig4.savefig('fig4BoostingELMPrecisionStats.png')
		# fig4.savefig('fig5MajorityELMPrecisionStats.png')
		fig4.savefig('fig6comparisonELMSVMPrecisionStats.png')

	def test_sig_boxplotAccuracyStyle(self):
		# print(self.acc.columns)
		fig, ax3 = plt.subplots(figsize=(9, 7))
		sns.set_style("whitegrid")
		ax3 = sns.boxplot(data=self.acc, color="white")
		ax3 = sns.swarmplot(data=self.acc, color=".25")
		# plt.ylabel("single ELMs Accuracy")
		# plt.ylabel("Bagging ELM Accuracy")
		# plt.ylabel("Boosting ELM Accuracy")
		# plt.ylabel("Majority ELM Accuracy")
		plt.ylabel("Accuracy Comparison")

		# statistical notation
		add_stat_annotation(ax3, data=self.acc,
							box_pairs=[("single ELMs", "majority voting ELMs"), ("single ELMs", "bagging-based ELMs"),
									   ("single ELMs", "boosting-based ELMs"), ("majority voting ELMs", "boosting-based ELMs"),
									   ("bagging-based ELMs", "boosting-based ELMs"), ("boosting-based ELMs", "SVMs")],
							test='t-test_ind', text_format='star', loc='inside', verbose=2)

		# plt.show(ax3)
		# fig.savefig('fig2SingleELMAccuracyStyleStats.png')
		# fig.savefig('fig3BaggingELMAccuracyStyleStats.png')
		# fig.savefig('fig4BoostingELMAccuracyStyleStats.png')
		# fig.savefig('fig5MajorityELMAccuracyStyleStats.png')
		fig.savefig('fig6comparisonELMSVMAccuracyStyleStats.png')

	def test_sig_boxplotRecall(self):
		# 	print(self.acc.columns)
		fig, ax3 = plt.subplots(figsize=(9, 7))
		sns.set_style("whitegrid")
		ax3 = sns.boxplot(data=self.recall, color="white")
		ax3 = sns.swarmplot(data=self.recall, color=".25")
		# plt.ylabel("single ELMs Recall")
		# plt.ylabel("Bagging ELM Recall")
		# plt.ylabel("Boosting ELM Recall")
		# plt.ylabel("Majority ELM Recall")
		plt.ylabel("Recall Comparison")

		# statistical notation
		add_stat_annotation(ax3, data=self.recall,
							box_pairs=[("single ELMs", "majority voting ELMs"), ("single ELMs", "bagging-based ELMs"),
									   ("single ELMs", "boosting-based ELMs"), ("majority voting ELMs", "boosting-based ELMs"),
									   ("bagging-based ELMs", "boosting-based ELMs"), ("boosting-based ELMs", "SVMs")],
							test='t-test_ind', text_format='star', loc='inside', verbose=2)

		# plt.show(ax3)
		# fig.savefig('fig2SingleELMRecallStats.png')
		# fig.savefig('fig3BaggingELMRecallStats.png')
		# fig.savefig('fig4BoostingELMRecallStats.png')
		# fig.savefig('fig5MajorityELMRecallStats.png')
		fig.savefig('fig6comparisonELMSVMRecallStats.png')

	# def test_sig_boxplotPrecision2(self):
	# 	print(self.acc.columns)
		# fig4, ax4 = plt.subplots(figsize=(9, 7))
		# sns.set_style("whitegrid")
		# ax4 = sns.boxplot(data=self.precision, color="white")
		# ax4 = sns.swarmplot(data=self.precision, color=".25")
		# plt.ylabel("precision")
		#
		# statistical notation
		# add_stat_annotation(ax4, data=self.precision,
		# 					box_pairs=[("10 models", "50 models"), ("10 models", "100 models"),
		# 							   ("10 models", "500 models")],
		# 					test='t-test_ind', text_format='star', loc='inside', verbose=2)
		#
		# plt.show(ax4)
		# fig4.savefig('fig1PrecisionNoBaselineStdFocusedStats.png')





