from unittest import TestCase
from SVM.hyperparameterTuningSVM import hyperparameterTuningSVM

class TestHyperparameterTuningProperties(TestCase):
	@classmethod
	def setUpClass(cls):
		cls.sampleX, cls.sampleT = hyperparameterTuningSVM('Data/data.csv', 'Data/targets.csv')

	def test_run(self):
		pass
		# print(self.sampleX)
		# print(self.sampleT)
		# print(self.sampleX.shape)
		# print(self.sampleT.shape)