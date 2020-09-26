import unittest
import os
from Single_ELMs.elmPopulationMetrics import elmPopulationMetrics

class TestElmPopulationMetrics(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.Xtr, cls.Xts, cls.Ttr, cls.Tts, cls.Y, cls.time, cls.acc, cls.recall, cls.precision = elmPopulationMetrics\
																  ("../Data/Datasets/npy_TrainTest_Std", 1000, "sigm")
		#cls.acc, cls.X, cls.T  = wc_cv("npy_TrainTest_Unstd", 1000, "sigm")
		#cls.acc, cls.X, cls.T  = wc_cv("npy_TrainTest_reducedStd", 1000, "sigm")
		#cls.acc, cls.X, cls.T  = wc_cv("npy_TrainTest_reducedUnstd", 1000, "sigm")
		# cls.acc, cls.X, cls.T = wc_cv("", 750, "sigm")

	def test_data_located(self):
		folder = os.path.join(os.path.dirname(__file__), "")
		var = os.path.join(folder, "train_x_0.npy")

	def test_cycling_through_data(self):
		pass

	def test_data_shapes(self):
		print (self.Xtr.shape)
		print(self.Xts.shape)
		print(self.Ttr.shape)
		print(self.Tts.shape)

	def test_training_speed(self):
		print(self.time)

	def test_single_trial_prediction(self):
		print(self.Y.shape)

	def test_metrics_calculations(self):
		print(self.acc)
		print(self.recall)
		print(self.precision)





