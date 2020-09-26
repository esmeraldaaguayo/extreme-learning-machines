import numpy as np


# FUNCTION splits data into training datasets and testing datasets for leave-one-out2subjtest subject
def oneSubjectOutCrossValidation(csv_file_x, csv_file_y):
	# X = np.loadtxt(csv_file_x, delimiter=',')
	# T = np.loadtxt(csv_file_y, delimiter=',')
	X = np.load(csv_file_x)
	T = np.load(csv_file_y)


	# divide data into train and test set
	indices = [np.arange(0, 4860), np.arange(4860, 9720), np.arange(9720, 14580), np.arange(14580, 19305),
			   np.arange(19305, 24165), np.arange(24165, 29025), np.arange(29025, 33885), np.arange(33885, 38745),
			   np.arange(38745, 43605), np.arange(43605, 48465), np.arange(48465, 53325), np.arange(53325, 58185),
			   np.arange(58185, 63045), np.arange(63045, 67770), np.arange(67770, 72630), np.arange(72630, 77490),
			   np.arange(77490, 82350), np.arange(82350, 87210)]

	# partition indices
	for i in range(18):
		items = [(i + j) % 18 for j in range(18)]
		print(items)
		idx_train = np.hstack([indices[j] for j in items[:-1]])
		idx_test = indices[items[-1]]

		train_x = X[idx_train]
		train_t = T[idx_train]
		test_x = X[idx_test]
		test_t = T[idx_test]

		print(train_x.shape)
		print(train_t.shape)
		print(test_x.shape)
		print(test_t.shape)

		# generate .npy files of data
		# np.save("Unstd_train_x_{}.npy".format(i), train_x)
		# np.save("Unstd_train_t_{}.npy".format(i), train_t)
		# np.save("Unstd_test_x_{}.npy".format(i), test_x)
		# np.save("Unstd_test_t_{}.npy".format(i), test_t)

		# np.save("Std_train_x_{}.npy".format(i), train_x)
		# np.save("Std_train_t_{}.npy".format(i), train_t)
		# np.save("Std_test_x_{}.npy".format(i), test_x)
		# np.save("Std_test_t_{}.npy".format(i), test_t)

		# np.save("reducedStd_train_x_{}.npy".format(i), train_x)
		# np.save("reducedStd_train_t_{}.npy".format(i), train_t)
		# np.save("reducedStd_test_x_{}.npy".format(i), test_x)
		# np.save("reducedStd_test_t_{}.npy".format(i), test_t)

		# np.save("reducedUnstd_train_x_{}.npy".format(i), train_x)
		# np.save("reducedUnstd_train_t_{}.npy".format(i), train_t)
		# np.save("reducedUnstd_test_x_{}.npy".format(i), test_x)
		# np.save("reducedUnstd_test_t_{}.npy".format(i), test_t)

		np.save("noBaselineStd_train_x_{}.npy".format(i), train_x)
		np.save("noBaselineStd_train_t_{}.npy".format(i), train_t)
		np.save("noBaselineStd_test_x_{}.npy".format(i), test_x)
		np.save("noBaselineStd_test_t_{}.npy".format(i), test_t)

		# np.save("noBaselineUnstd_train_x_{}.npy".format(i), train_x)
		# np.save("noBaselineUnstd_train_t_{}.npy".format(i), train_t)
		# np.save("noBaselineUnstd_test_x_{}.npy".format(i), test_x)
		# np.save("noBaselineUnstd_test_t_{}.npy".format(i), test_t)
	return X, T