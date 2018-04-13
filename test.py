'''
@author Andrea Corriga
@contact me@andreacorriga.com
@date 2018
@version 1.0
'''
import time
import argparse
import cv2
from PIL import Image
from sklearn import svm

# Import my alghorithms
from algorithms.LBP import LBP
# Import my utils method
from utils.utils import *
from utils.dataset import *

from sklearn.metrics import accuracy_score #c alculate accuracy
from sklearn.externals import joblib # save and load model
from sklearn.model_selection import train_test_split # in order to split training and test
import numpy
from skimage.feature import local_binary_pattern

def main():

	parser = argparse.ArgumentParser(description='Run the local binary patterns algorithm using either a single process or multiple processes.')
	parser.add_argument('--dataset', dest='dataset', type=str, default='YaleFaces', help='Main folder of the dataset')
	parser.add_argument('--algorithm', dest='algorithm', type=str, default='lbp', help='Algorithm to use: "lbp" or "elbp"')
	parser.add_argument('--training', dest='training', action='store_true', default=False, help='whether or not an output image should be produced')
	parser.add_argument('--histEq', dest='histEq', action='store_true', default=False, help='if you want to equialize the histogram before calculating LBP or ELBP')
	parser.add_argument('--output', dest='output', action='store_true', default=False, help='if you want to save the png of LBP image')

	arguments = parser.parse_args()
	datasetMainFolder = os.getcwd() + "/datasets/"

	# Security check about the algorithm
	if(arguments.algorithm != "lbp" and arguments.algorithm != "elbp"):
		print("Algorithm not valid. Choose between LBP or ELBP")
		return

	# Security check about the dataset
	if os.path.isdir(datasetMainFolder + arguments.dataset) == False:
		print('The Dataset "' + arguments.dataset + '" doesn\'t exist')
		return

	# Helpful instad of write datasetMainFolder + arguments.dataset + "/"
	datasetFolder = datasetMainFolder + arguments.dataset + "/"

	# Get Dataset information
	classes, filename, xFilepaths, y = getDataset(arguments.dataset)
	x = []

	print("Launching " + arguments.algorithm.upper() + " algorithm on the " + arguments.dataset + " dataset...")
	startTime = time.time()

	# This counter is used to store the png 
	counter = 0

	# if --output is passed as parameter
	if arguments.output == True:
		createFolder(arguments.dataset, arguments.algorithm.upper() )


	for xfp in xFilepaths:
		img = imgRead(datasetFolder + xfp)

		#imgShow(numpy.matrix(img))

		# Check if img exist (security check)
		if img:
			# if --histEq is passed as parameter, perform an histogram equalization
			if(arguments.histEq == True):
				img =  histogramEqualization(img) 

			lbp_value = local_binary_pattern(img, 8, 1, method='uniform')

			# Split img into 12*12 blocks
			shaped = blockshaped(lbp_value, 16, 14)

			xBlocks = []
			for s in shaped:
				xBlocks.append(getHistogram(s))

			x.append(numpy.concatenate(xBlocks))

			# if --output is passed as parameter
			if arguments.output == True:
				saveImgFromArray(lbpObject.getImage(), arguments.dataset, filenames[counter], arguments.algorithm.upper() )

		# If the image doens't exist
		else:
			print("The image: " + datasetFolder + xfp + " doesn't exist")	
		# Add counter for new image
		counter = counter + 1


	print("--- " + arguments.algorithm.upper() + " done in %s seconds ---" % (time.time() - startTime))

	print("Split dataset into training and test set [0.77] [0.33]")

	# Split dataset x (feature vector) and y (label) into training and test set
	xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.33)
	
	print("Launching SVM...")

	startTime = time.time()

	# if --training is passet as parameter, perform the training of model
	if arguments.training == True:
		trainingTime = time.time()
		clf = svm.LinearSVC()
		print("Start training...")
		clf.fit(xTrain, yTrain)
		joblib.dump(clf, 'model/svm_block.pkl') 
		print("--- Training done in %s seconds ---" % (time.time() - trainingTime))

	# Test the model
	clf = joblib.load('model/svm_block.pkl') 
	print("Start testing...")
	predicted = clf.predict(xTest)

	print("--- SVM done in %s seconds ---" % (time.time() - startTime))

	print("Accuracy: " + str(accuracy_score(yTest, predicted)))
	
if __name__ == "__main__":
	main()



