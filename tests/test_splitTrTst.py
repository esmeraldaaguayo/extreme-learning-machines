import unittest
import numpy as np
from Data.Datasets.splitTrTst import split_train_test


class TestSplitTrTst(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		#cls.x, cls.t = split_train_test('x2DUnStd.csv','y2D.csv')
		#cls.x, cls.t = split_train_test('x2DreducedStd.csv', 'y2D.csv')
		# cls.x, cls.t = split_train_test('x2DreducedUnStd.csv', 'y2D.csv')
		# cls.x, cls.t = split_train_test('../Data/Raw/x2DnoBaselineStd.csv', '../Data/Raw/y2D.csv')
		cls.x, cls.t = split_train_test('../Data/Raw/x2DnoBaselineUnStd.csv', '../Data/Raw/y2D.csv')

	def test_load(self):
		self.assertIsInstance(self.x, np.ndarray)
		self.assertIsInstance(self.t, np.ndarray)
