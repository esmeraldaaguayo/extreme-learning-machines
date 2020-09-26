import unittest
import numpy as np
from Single_ELMs.elmWc import wc_cv
import os


class TestElmWcCv(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		#cls.acc, cls.X, cls.T  = wc_cv("npy_TrainTest_Std", 1000, "sigm")
		#cls.acc, cls.X, cls.T  = wc_cv("npy_TrainTest_Unstd", 1000, "sigm")
		#cls.acc, cls.X, cls.T  = wc_cv("npy_TrainTest_reducedStd", 1000, "sigm")
		#cls.acc, cls.X, cls.T  = wc_cv("npy_TrainTest_reducedUnstd", 1000, "sigm")
		cls.acc, cls.X, cls.T = wc_cv("", 750, "sigm")

	def test_data_located(self):
		folder = os.path.join(os.path.dirname(__file__), "")
		var = os.path.join(folder, "train_x_0.npy")

	def test_data_loaded(self):
		self.assertIsInstance(self.X, np.ndarray)

	def test_accuracy(self):
		target = 0.5
		self.assertGreater(self.acc, target)

	def test_speed(self):
		print("\n training took: %f seconds" % self.T)