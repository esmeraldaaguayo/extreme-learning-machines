from unittest import TestCase
from ensembleELM.revertSampleWeighting import revertSampleWeighting
import numpy as np

class TestRevertnessProperties(TestCase):
	@classmethod
	def setUpClass(cls):
		cls.samplecT = revertSampleWeighting('Data/weighteddata.csv', 'Data/weightedVector.csv')

	def test_run(self):
		print(self.samplecT)
		# print(self.sampleT.shape)