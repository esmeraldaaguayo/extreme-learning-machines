import unittest
import numpy as np
from Data.Datasets.splitTrainValTest import leave_one_out


class TestLeaveOneOut(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		#cls.x, cls.t, cls.indices, cls.test_x, cls.val_x, cls.train_x = leave_one_out('x2D.csv', 'y.csv')
		# cls.x, cls.t = leave_one_out('x2D.csv', 'y.csv')
		# cls.x, cls.t = leave_one_out('../Data/Raw/x2DUnStd.csv', '../Data/Raw/y2D.csv')
		cls.x, cls.t = leave_one_out('../Data/Raw/x2DStd.csv', '../Data/Raw/y2D.csv')

	def test_load(self):
		# test that the files are loaded into np arrays
		self.assertEqual(isinstance(self.x, np.ndarray), True)
		self.assertEqual(isinstance(self.t, np.ndarray), True)

	def test_epoch_number_per_subject(self):
		# check that total epochs per subject are correct
		#self.assertEqual(len(self.indices[0]), 4860)
		#self.assertEqual(len(self.indices[1]), 4860)
		#self.assertEqual(len(self.indices[2]), 4860)
		#self.assertEqual(len(self.indices[3]), 4860)
		#self.assertEqual(len(self.indices[4]), 4725)
		#self.assertEqual(len(self.indices[5]), 4860)
		#self.assertEqual(len(self.indices[6]), 4860)
		#self.assertEqual(len(self.indices[7]), 4860)
		#self.assertEqual(len(self.indices[8]), 4860)
		#self.assertEqual(len(self.indices[9]), 4860)
		#self.assertEqual(len(self.indices[10]), 4860)
		#self.assertEqual(len(self.indices[11]), 4860)
		#self.assertEqual(len(self.indices[12]), 4860)
		#self.assertEqual(len(self.indices[13]), 4860)
		#self.assertEqual(len(self.indices[14]), 4860)
		#self.assertEqual(len(self.indices[15]), 4725)
		#self.assertEqual(len(self.indices[16]), 4860)
		#self.assertEqual(len(self.indices[17]), 4860)
		#self.assertEqual(len(self.indices[18]), 4860)
		#self.assertEqual(len(self.indices[19]), 4860)
		pass


	def test_partition_shapes(self):
		# test correct allocation of data to train, validation and test sets
		#self.assertEqual(self.test_x, (4860, 2240))
		#self.assertEqual(self.val_x, (4860, 2240))
		#self.assertEqual(self.train_x, (87210, 2240))
		pass


