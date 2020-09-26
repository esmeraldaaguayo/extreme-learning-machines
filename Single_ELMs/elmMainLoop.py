import pandas as pd
from Single_ELMs.auxELMUnstd import auxELMUnstd
from Single_ELMs.auxELMUnstdReduced import auxELMUnstdReduced
from Single_ELMs.auxELMUnstdNoBaseline import auxELMUnstdNoBaseline
from Single_ELMs.auxELMStd import auxELMStd
from Single_ELMs.auxELMStdReduced import auxELMStdReduced
from Single_ELMs.auxELMStdNoBaseline import auxELMStdNoBaseline

def elmMainLoop(folder1, folder2, folder3, folder4, folder5, folder6):

	accuracyUnstd, recallUnstd, precisionUnstd = \
		auxELMUnstd(folder1, 1000, "sigm")
		# auxELMUnstd(folder1, "../Data/Datasets/globalSplit/unstd", 1000, "sigm")
	accuracyUnstdReduced, recallUnstdReduced, precisionUnstdReduced = \
		auxELMUnstdReduced(folder2, 1000, "sigm")
		# auxELMUnstdReduced(folder2, "../Data/Datasets/globalSplit/reducedUnstd", 1000, "sigm")
	accuracyUnstdNoBaseline, recallUnstdNoBaseline, precisionUnstdNoBaseline = \
		auxELMUnstdNoBaseline(folder3, 1000, "sigm")
		# auxELMUnstdNoBaseline(folder3, "../Data/Datasets/globalSplit/noBaselineUnstd", 1000, "sigm")

	accuracyStd, recallStd, precisionStd = \
		auxELMStd(folder4, 1000, "sigm")
		# auxELMStd(folder4, "../Data/Datasets/globalSplit/std", 1000, "sigm")
	accuracyStdReduced, recallStdReduced, precisionStdReduced = \
		auxELMStdReduced(folder5, 1000, "sigm")
		# auxELMStdReduced(folder5, "../Data/Datasets/globalSplit/reducedStd", 1000, "sigm")
	accuracyStdNoBaseline, recallStdNoBaseline, precisionStdNoBaseline = \
		auxELMStdNoBaseline(folder6, 1000, "sigm")
		# auxELMStdNoBaseline(folder6, "../Data/Datasets/globalSplit/noBaselineStd", 1000, "sigm")

	dictAccuracy = {'Unstd':accuracyUnstd, 'Reduced Unstd':accuracyUnstdReduced, 'No Baseline Unstd': accuracyUnstdNoBaseline,
			'Std':accuracyStd, 'Reduced Std':accuracyStdReduced, 'No Baseline Std':accuracyStdNoBaseline}
	dictRecall = {'Unstd': recallUnstd, 'Reduced Unstd': recallUnstdReduced, 'No Baseline Unstd': recallUnstdNoBaseline,
					'Std': recallStd, 'Reduced Std': recallStdReduced, 'No Baseline Std': recallStdNoBaseline}
	dictPrecision = {'Unstd': precisionUnstd, 'Reduced Unstd': precisionUnstdReduced, 'No Baseline Unstd': precisionUnstdNoBaseline,
					'Std': precisionStd, 'Reduced Std': precisionStdReduced, 'No Baseline Std': precisionStdNoBaseline}

	df_accuracy = pd.DataFrame(dictAccuracy)
	df_recall = pd.DataFrame(dictRecall)
	df_precision = pd.DataFrame(dictPrecision)


	return df_accuracy, df_recall, df_precision