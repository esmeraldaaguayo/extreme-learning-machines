import numpy as np
import pandas as pd

# data = pd.read_csv('Single_ELMs/out2subjtest/metrics/noBaselineStd_metrics.csv', delimiter=',')
# data = pd.read_csv('Single_ELMs/outCV/metrics/Unstd_CV_metrics.csv', delimiter=',')
# data = pd.read_csv('Single_ELMs/outCV/metrics/reducedUnstd_CV_metrics.csv', delimiter=',')
# data = pd.read_csv('Single_ELMs/outCV/metrics/noBaselineUnstd_CV_metrics.csv', delimiter=',')
# data = pd.read_csv('Single_ELMs/outCV/metrics/Std_CV_metrics.csv', delimiter=',')
# data = pd.read_csv('Single_ELMs/outCV/metrics/reducedStd_CV_metrics.csv', delimiter=',')
# data = pd.read_csv('Single_ELMs/outCV/metrics/noBaselineStd_CV_metrics.csv', delimiter=',')
# data = pd.read_csv('ensembleELM/outCV/generalMajorityVoting/model100/majorityELM_voted_metrics_100_model.csv', delimiter=',')
# data = pd.read_csv('ensembleELM/outCV/generalMajorityVoting/model50/majorityELM_voted_metrics_50_model.csv', delimiter=',')
# data = pd.read_csv('ensembleELM/outCV/generalMajorityVoting/model10/majorityELM_voted_metrics_10_model.csv', delimiter=',')
# data = pd.read_csv('ensembleELM/outCV/baggingELM/model100/baggingELM_voted_metrics_100_model.csv', delimiter=',')
# data = pd.read_csv('ensembleELM/outCV/baggingELM/model50/baggingELM_voted_metrics_50_model.csv', delimiter=',')
# data = pd.read_csv('ensembleELM/outCV/baggingELM/model10/baggingELM_voted_metrics_10_model.csv', delimiter=',')
data = pd.read_csv('ensembleELM/outCV/boostingELM/model100/boostingELM_voted_metrics_100_model.csv', delimiter=',')
# data = pd.read_csv('ensembleELM/outCV/boostingELM/model50/boostingELM_voted_metrics_50_model.csv', delimiter=',')
# data = pd.read_csv('ensembleELM/outCV/boostingELM/model10/boostingELM_voted_metrics_10_model.csv', delimiter=',')

# sum = data.sum(axis=0)
# acc = sum[0]/18
# recal = sum[1]/18
# prec = sum[2]/18
# time = sum[3]/18
# stdevTime =
# print(acc)
# print(recal)
# print(prec)
# print(time)
print(data.mean())
print(data.std())