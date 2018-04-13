import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from utils.utils import *
from utils.dataset import *
from sklearn.model_selection import train_test_split # in order to split training and test
import numpy
from skimage.feature import local_binary_pattern
from algorithms.LBP import LBP

xFilepaths = getPersonalDataset()

x = []
for xfp in xFilepaths:
	img = imgRead(xfp)

	# Check if img exist (security check)
	if img:

		#img =  histogramEqualization(img) 

		lbp_value = local_binary_pattern(img, 8, 1)

		# Split img into 12*12 blocks
		shaped = blockshaped(lbp_value, 60, 60)

		xBlocks = []
		for s in shaped:
			xBlocks.append(getHistogram(s))

		x.append(numpy.concatenate(xBlocks))

xTrain = x[:5]
xTest = x[4:5]


# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(x)

y_pred = clf.predict(x)

print(y_pred)
'''
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

print(y_pred_train)
'''