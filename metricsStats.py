import pandas as pd
from scipy import stats
import numpy as np

def metricsStats(x1,x2,x3,x4,x5): #,x6):
	x1 = pd.read_csv(x1)
	x2 = pd.read_csv(x2)
	x3 = pd.read_csv(x3)
	x4 = pd.read_csv(x4)
	x5 = pd.read_csv(x5)
	# x6 = pd.read_csv(x6)

	accuracy = [x1["accuracy"], x2["accuracy"], x3["accuracy"], x4["accuracy"], x5["accuracy"]] #, x6["accuracy"]]
	recall = [x1["recall"], x2["recall"], x3["recall"], x4["recall"], x5["recall"]] #, x6["recall"]]
	precision = [x1["precision"], x2["precision"], x3["precision"], x4["precision"], x5["precision"]] #, x6["precision"]]

	# headers = ["Unstd", "Reduced Unstd", "no Baseline Unstd", "Std", "Reduced Std", "no Baseline Std"]
	# headers = ["10 models", "50 models", "100 models", "500 models"]
	# headers = ["10 models", "50 models", "100 models"]
	headers = ["single ELMs", "majority voting ELMs", "bagging-based ELMs", "boosting-based ELMs", "SVMs"]

	df_accuracy = pd.concat(accuracy,axis=1,keys=headers)
	df_recall = pd.concat(recall, axis=1, keys=headers)
	df_precision = pd.concat(precision, axis=1, keys=headers)

	df_ttest_sig_accuracy = pd.DataFrame({'accuracy index':['t statistic', 'p statistic']})

	for item1 in headers:
		for item2 in headers:
			if item1 != item2:
				t, p = stats.ttest_ind(df_accuracy[item1], df_accuracy[item2])
				if p <= 0.05:
					header = item1 + item2
					dict = {header: [t, p]}
					df_dict = pd.DataFrame(dict)
					df_ttest_sig_accuracy = pd.concat([df_ttest_sig_accuracy,df_dict], axis=1)

	df_ttest_sig_recall = pd.DataFrame({'recall index': ['t statistic', 'p statistic']})

	for item1 in headers:
		for item2 in headers:
			if item1 != item2:
				t, p = stats.ttest_ind(df_recall[item1], df_recall[item2])
				if p <= 0.05:
					header = item1 + item2
					dict = {header: [t, p]}
					df_dict = pd.DataFrame(dict)
					df_ttest_sig_recall = pd.concat([df_ttest_sig_recall, df_dict], axis=1)

	df_ttest_sig_precision = pd.DataFrame({'precision index': ['t statistic', 'p statistic']})

	for item1 in headers:
		for item2 in headers:
			if item1 != item2:
				t, p = stats.ttest_ind(df_precision[item1], df_precision[item2])
				if p <= 0.05:
					header = item1 + item2
					dict = {header: [t, p]}
					df_dict = pd.DataFrame(dict)
					df_ttest_sig_precision = pd.concat([df_ttest_sig_precision, df_dict], axis=1)

	# save csv of metric ttest statistics
		df_ttest_sig_accuracy.to_csv("sig_ttest_accuracy_metrics.csv", index=False)
		df_ttest_sig_recall.to_csv("sig_ttest_recall_metrics.csv", index=False)
		df_ttest_sig_precision.to_csv("sig_ttest_precision_metrics.csv", index=False)

	# do one-way ANOVA
	# anova_statistics = stats.f_oneway(df_accuracy["Unstd"], df_accuracy["Reduced Unstd"], df_accuracy["no Baseline Unstd"],
	# 								  df_accuracy["Std"], df_accuracy["Reduced Std"], df_accuracy["no Baseline Std"])
	# anova_statistics = stats.f_oneway(df_accuracy["10 models"], df_accuracy["50 models"],
	# 								  df_accuracy["100 models"], df_accuracy["500 models"])
	anova_statistics = stats.f_oneway(df_accuracy["single ELMs"], df_accuracy["majority voting ELMs"],
									  df_accuracy["bagging-based ELMs"], df_accuracy["boosting-based ELMs"],
									  df_accuracy["SVMs"])
	print(anova_statistics)
	return df_accuracy, df_recall, df_precision, df_ttest_sig_accuracy, df_ttest_sig_recall, df_ttest_sig_precision