from unittest import TestCase
from ensembleELM.baggingELM import baggingELM

class TestBaggingProperties(TestCase):
	@classmethod
	def setUpClass(cls):
		cls.predictions, cls.actual, cls.time = \
			baggingELM('../Data/Datasets/globalSplit/Xtr_noBaselineStd.csv',
					   '../Data/Datasets/globalSplit/Xts_noBaselineStd.csv',
					   '../Data/Datasets/globalSplit/Ttr_noBaselineStd.csv',
					   '../Data/Datasets/globalSplit/Tts_noBaselineStd.csv')

	def test_output(self):
		print(self.predictions.shape)
		print(self.actual)
		print(self.time)
