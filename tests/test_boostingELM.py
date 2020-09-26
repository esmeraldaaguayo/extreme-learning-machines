from unittest import TestCase
from ensembleELM.boostingELM import boostingELM

class TestBoostingProperties(TestCase):
	@classmethod
	def setUpClass(cls):
		cls.acc, cls.recall, cls.precision = \
			boostingELM('../Data/Datasets/globalSplit/Xtr_std.csv',
					   '../Data/Datasets/globalSplit/Xts_std.csv',
					   '../Data/Datasets/globalSplit/Ttr_std.csv',
					   '../Data/Datasets/globalSplit/Tts_std.csv')

	def test_metrics(self):
		print(self.acc)
		print(self.recall)
		print(self.precision)