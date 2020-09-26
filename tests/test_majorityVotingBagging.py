from unittest import TestCase
import os
from ensembleELM.majorityVotingBagging import majorityVotingBagging

class TestmajorityVoting(TestCase):
	@classmethod
	def setUpClass(cls):
		cls.metrics = majorityVotingBagging('Data/data.csv', 'Data/targets.csv')

	def test_run(self):
		pass

	def test_metricValues(self):
		pass
		# print(self.acc)
		# print(self.recall)
		# print(self.precision)


