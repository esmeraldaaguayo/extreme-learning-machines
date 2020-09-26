from unittest import TestCase
from ensembleELM.instanceWeighting import instanceWeighting
import numpy as np

class TestBoostingProperties(TestCase):
	@classmethod
	def setUpClass(cls):
		cls.sampleX, cls.sampleT = instanceWeighting('Data/data.csv', 'Data/targets.csv', 'Data/vector.csv')

	def test_run(self):
		print(self.sampleX)
		print(self.sampleT)
		# print(self.sampleX.shape)
		# print(self.sampleT.shape)
