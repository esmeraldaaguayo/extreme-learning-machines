import unittest
import os
from Single_ELMs.auxELMUnstd import auxELMUnstd

class TestauxELMUnstd(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.acc, cls.recall, cls.precision = auxELMUnstd("../Data/Datasets/npy_TrainTest_Unstd", 1000, "sigm")

	def test_data_located(self):
		folder = os.path.join(os.path.dirname(__file__), "")
		var = os.path.join(folder, "Unstd_train_x_0.npy")

	def test_cycling_through_data(self):
		pass

	def test_metrics_calculations(self):
		print(self.acc)
		print(self.recall)
		print(self.precision)

