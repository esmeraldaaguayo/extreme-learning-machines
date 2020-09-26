import unittest
from Single_ELMs.elmPythonStdWc import weightedClassification
import os


class TestElmWcCv(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.acc, cls.X = weightedClassification("", 1000, "sigm")

	def test_data_located(self):
		folder = os.path.join(os.path.dirname(__file__), "")
		var = os.path.join(folder, "train_x_0.npy")