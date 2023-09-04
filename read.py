import pandas
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_file = '/root/projects/myself/runs/dataset37/mynet_ablation/exp_2023-08-13_22:49/log.csv'
# results/log.csv

ignore = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
results = pandas.read_csv('%s' % log_file, usecols=[1, 2, 3, 4, 5, 6], skiprows=ignore)
diceMax = pandas.read_csv(log_file, usecols=[1])
haduassMin = pandas.read_csv(log_file, usecols=[2])
iouMax = pandas.read_csv(log_file, usecols=[3])
precisionMax = pandas.read_csv(log_file, usecols=[4])
recallMax = pandas.read_csv(log_file, usecols=[5])
accuracyMax = pandas.read_csv(log_file, usecols=[6])
# print(results)

print("Mean: ", np.mean(results, axis=0))
print("\nstd:  ", np.std(results, axis=0))
print("\n***--- diceMax:", np.max(diceMax, axis=0))
print("***---haduassMin:", np.min(haduassMin, axis=0))
print("***---iouMax:", np.max(iouMax, axis=0))
print("***---precisionMax:", np.max(precisionMax, axis=0))
print("***---recallMax:", np.max(recallMax, axis=0))
print("***---accuracyMax:", np.max(accuracyMax, axis=0))
