from unittest import TestCase
from ensembleELM.bootstrapAggregating import bootstrapAggregating

class TestBaggingProperties(TestCase):
	@classmethod
	def setUpClass(cls):
		cls.sampleX, cls.sampleT = bootstrapAggregating('Data/data.csv', 'Data/targets.csv')

	def test_output(self):
		print(self.sampleX.shape)
		print(self.sampleT.shape)

	def test_randomization(self):
		print(self.sampleX[1])


