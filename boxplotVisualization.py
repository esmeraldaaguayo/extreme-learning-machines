from metricsStats import metricsStats
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation

def boxplotVisualization(df_metrics):
	df_metrics = df_metrics
	m = df_metrics.shape[0]

	fig1, ax1 = plt.subplots(figsize=(9, 7))
	sns.set_style("whitegrid")
	ax1 = sns.boxplot(data=df_metrics['accuracy'], color="white")
	ax1 = sns.swarmplot(data=df_metrics['accuracy'], color=".25")
	plt.ylabel("Accuracy")
	fig1.savefig('fig3Accuracy_{}.png'.format(m))

	fig2, ax2 = plt.subplots(figsize=(9, 7))
	sns.set_style("whitegrid")
	ax2 = sns.boxplot(data=df_metrics['accuracy'], color="white")
	ax2 = sns.swarmplot(data=df_metrics['accuracy'], color=".25")
	plt.ylabel("Accuracy")
	fig2.savefig('fig3AccuracyStyle_{}.png'.format(m))

	fig3, ax3 = plt.subplots(figsize=(9, 7))
	sns.set_style("whitegrid")
	ax3 = sns.boxplot(data=df_metrics['recall'], color="white")
	ax3 = sns.swarmplot(data=df_metrics['recall'], color=".25")
	plt.ylabel("Recall")
	fig3.savefig('fig3Recall_{}.png'.format(m))

	fig4, ax4 = plt.subplots(figsize=(9, 7))
	sns.set_style("whitegrid")
	ax4 = sns.boxplot(data=df_metrics['precision'], color="white")
	ax4 = sns.swarmplot(data=df_metrics['precision'], color=".25")
	plt.ylabel("Precision")
	fig4.savefig('fig3Precision_{}.png'.format(m))
	return