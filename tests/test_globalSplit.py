import unittest
from Data.Datasets.globalSplit import globalSplit

class TestSplitAccuracy(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.X, cls.T, cls.Xtr, cls.Xts, cls.Ttr, cls.Tts = \
			globalSplit('../Data/Raw/x2DnoBaselineStd.csv', '../Data/Raw/y2D.csv')
			# globalSplit('data/data.csv', 'data/targets.csv')


	def test_originalSizes(self):
		pass
		# self.assertEqual(self.X.shape, (30,3))
		# self.assertEqual(self.T.shape, (30, 2))
		# self.assertEqual(self.X.shape, (96930, 2240))
		# self.assertEqual(self.T.shape, (96930, 2))

	def test_stratify_efficacy(self):
		pass
		# print(self.Xtr)
		# print(self.Xts)
		# print(self.Ttr)
		# print(self.Tts)
		# assert same ratio
		# self.assertEqual(self.Ttr.shape, (27, 2))
		# self.assertEqual(self.Tts.shape, (3, 2))
		# print(self.Ttr.groupby(0).count())
		# print(self.Tts.groupby(0).count())
		# self.assertEqual(self.Ttr.shape, (87237, 2))
		# self.assertEqual(self.Tts.shape, (9693, 2))
		# print(self.Ttr.groupby(0).count())
		# print(self.Tts.groupby(0).count())

