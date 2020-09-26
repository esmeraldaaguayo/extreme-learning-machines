from unittest import TestCase
from ensembleELM.majorityVotingBoosting import majorityVotingBoosting


class TestmajorityVoting(TestCase):
	@classmethod
	def setUpClass(cls):
		cls.metrics, cls.acc, cls.recall, cls.prec = majorityVotingBoosting([1,1], [2,3], 4)

	def test_run(self):
		pass