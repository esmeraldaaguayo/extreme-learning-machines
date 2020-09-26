import hpelm
import numpy as np
import hpelm.modules
import csv

# FUNCTION ELM hyperparameter tunning

model = hpelm.HPELM(2240, 2, classification='wc') # wc uses [9. 1.125] as w
model.add_neurons(500, 'sigm')


# compute HH and HT matrices
model.add_data('train_x_0.hdf5', 'train_t_0.hdf5', fHH='HH0.hdf5', fHT='HT0.hdf5')
l, e, m = model.validation_corr('HH0.hdf5', 'HT0.hdf5', 'validation_x_0.hdf5', 'validation_t_0.hdf5', steps=100)


with open('confusion0.txt', 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(m)

csvData = []
csvData.append(l)
csvData.append(e)
with open('dataset0.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)
csvFile.close()


#list = []
#for i in range(2):
	#model.add_data('train_x_%d.hdf5'%i, 'train_t_%d.hdf5'%i, istart=0, fHH='HH%d.hdf5'%i, fHT='HT%d.hdf5'%i)
	#l, e, m = model.validation_corr('HH%d.hdf5'%i, 'HT%d.hdf5'%i, 'validation_x_%d.hdf5'%i, 'validation_t_%d.hdf5'%i, steps=100)
	#list.append([l,e,m])
	#print('done')