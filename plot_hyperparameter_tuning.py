import matplotlib.pyplot as plt
import numpy as np
import csv
# FUNCTION  plot hyperparameter tuning data

data_string_mix_dict = {}

for idx in range(20):
	with open('dataset%d.csv' % idx, 'r') as f:
		contents = list(csv.reader(f, delimiter=','))
		data_string_mix_dict[idx] = contents

data_digits_dict = {}
for idx in range(20):
	list = []
	x = np.array(data_string_mix_dict[idx][0], dtype=np.int)
	list.append(x)
	temp = np.array(data_string_mix_dict[idx][1], dtype=np.float)
	y = np.subtract(1, temp)
	list.append(y)
	data_digits_dict[idx] = list

fig = plt.figure()
ax = plt.axes()

# ax.plot(x,y)

plt.plot(data_digits_dict[0][0], data_digits_dict[0][1])
plt.plot(data_digits_dict[1][0], data_digits_dict[1][1])
plt.plot(data_digits_dict[2][0], data_digits_dict[2][1])
plt.plot(data_digits_dict[3][0], data_digits_dict[3][1])
plt.plot(data_digits_dict[4][0], data_digits_dict[4][1])
plt.plot(data_digits_dict[5][0], data_digits_dict[5][1])
plt.plot(data_digits_dict[6][0], data_digits_dict[6][1])
plt.plot(data_digits_dict[7][0], data_digits_dict[7][1])
plt.plot(data_digits_dict[8][0], data_digits_dict[8][1])
plt.plot(data_digits_dict[9][0], data_digits_dict[9][1])
plt.plot(data_digits_dict[10][0], data_digits_dict[10][1])
plt.plot(data_digits_dict[11][0], data_digits_dict[11][1])
plt.plot(data_digits_dict[12][0], data_digits_dict[12][1])
plt.plot(data_digits_dict[13][0], data_digits_dict[13][1])
plt.plot(data_digits_dict[14][0], data_digits_dict[14][1])
plt.plot(data_digits_dict[15][0], data_digits_dict[15][1])
plt.plot(data_digits_dict[16][0], data_digits_dict[16][1])
plt.plot(data_digits_dict[17][0], data_digits_dict[17][1])
plt.plot(data_digits_dict[18][0], data_digits_dict[18][1])
plt.plot(data_digits_dict[19][0], data_digits_dict[19][1])
#plt.plot(data_digits_dict[0][0], data_digits_dict[0][1], label='data0....19')

plt.legend(frameon=False, loc='lower center', ncol=5)
plt.title("Hyperparameter Tuning")
plt.xlabel("Number of Hidden Neurons")
plt.ylabel("Accuracy of Validation")
plt.xlim(-100, 1250)
# plt.ylim(-1.5, 1.5)
plt.show()
