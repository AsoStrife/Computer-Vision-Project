'''
@author Andrea Corriga
@contact me@andreacorriga.com
@date 2018
@version 1.0
'''

import cv2, sys, numpy, os, argparse
from skimage.feature import local_binary_pattern

"""
Return an array of shape (n, nrows, ncols) where
n * nrows * ncols = arr.size

If arr is a 2D array, the returned array should look like n subblocks with
each subblock preserving the "physical" layout of arr.
"""
def blockshaped(arr, nrows, ncols):

	h, w = arr.shape
	return (arr.reshape(h//nrows, nrows, -1, ncols)
			.swapaxes(1,2)
			.reshape(-1, nrows, ncols))

# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html
def getHistogram(imgArray):
	hist, bin_edges = numpy.histogram(imgArray, density=True)
	return hist


# Perform LBP with multiblock
def LBP(img): 
	lbp_value = local_binary_pattern(img, 8, 1)

	# Split img into 10*10 blocks
	shaped = blockshaped(lbp_value, 10, 13)

	# Calculate the histogram for each block
	xBlocks = []
	for s in shaped:
		xBlocks.append(getHistogram(s))

	return numpy.concatenate(xBlocks)

# Get datasets photos
def getDatasets(datasets):
	(images, lables, names, id) = ([], [], {}, 0)
	
	for (subdirs, dirs, files) in os.walk(datasets):
		for subdir in dirs:
			names[id] = subdir
			subjectpath = os.path.join(datasets, subdir)
			for filename in os.listdir(subjectpath):
				path = subjectpath + '/' + filename
				lable = id
				images.append(cv2.imread(path, 0))
				lables.append(int(lable))
			id += 1

	return images, lables, names, id