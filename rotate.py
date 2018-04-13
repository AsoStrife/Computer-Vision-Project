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

def main():

	parser = argparse.ArgumentParser(description='Run the local binary patterns algorithm using either a single process or multiple processes.')
	parser.add_argument('--dataset', dest='dataset', type=str, default='YaleFaces', help='Main folder of the dataset')
	parser.add_argument('--histEq', dest='histEq', action='store_true', default=False, help='if you want to equialize the histogram before calculating LBP or ELBP')

	arguments = parser.parse_args()
	datasetMainFolder = os.getcwd() + "/datasets/"

	# Security check about the dataset
	if os.path.isdir(datasetMainFolder + arguments.dataset) == False:
		print('The Dataset "' + arguments.dataset + '" doesn\'t exist')
		return

	# Helpful instad of write datasetMainFolder + arguments.dataset + "/"
	datasetFolder = datasetMainFolder + arguments.dataset + "/"

	# Get Dataset information
	classes, filenames, xFilepaths, y = getDataset(arguments.dataset)

	print("Launching rotate algorithm algorithm for the " + arguments.dataset + " dataset...")
	startTime = time.time()

	# This counter is used to store the png 
	counter = 0

	createFolder(arguments.dataset + "_rotate")

	for yfolder in y:
		createFolder(arguments.dataset + "_rotate/" + yfolder)

	for xfp in xFilepaths:
		img = imgRead(datasetFolder + xfp)
		# Check if img exist (security check)
		if img:
			# if --histEq is passed as parameter, perform an histogram equalization
			if(arguments.histEq == True):
				img =  histogramEqualization(img) 

			imgRotate = rotateImg(getImgArray(img), 90)

			rotateObj = getImgObjFromArray(imgRotate)

			saveImgFromArray(rotateObj, arguments.dataset + "_rotate/" + y[counter], filenames[counter] )
		# If the image doens't exist
		else:
			print("The image: " + datasetFolder + xfp + " doesn't exist")	
		
		counter = counter + 1

	print("--- Rotate done in %s seconds ---" % (time.time() - startTime))

	
if __name__ == "__main__":
	main()



