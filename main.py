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
	parser.add_argument('--algorithm', dest='algorithm', type=str, default='lbp', help='Algorithm to use: "lbp" or "elbp"')
	parser.add_argument('--training', dest='training', action='store_true', default=False, help='whether or not an output image should be produced')

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

	datasetFolder = datasetMainFolder + arguments.dataset + "/"

	classes, xFilepath, y = getDataset(arguments.dataset)
	x = []

	print("Launching " + arguments.algorithm.upper() + " algorithm on the " + arguments.dataset + " dataset...")
	startTime = time.time()

	for xfp in xFilepath:
		img = imgRead(datasetFolder + xfp)
		# Check if img exist
		if img:
			# Check if the img contain a valid face and execute the LBP class
			lbpObject = LBP(img)
			lbpObject.execute()
			x.append(lbpObject.getImageArray())
		else:
			print("The image: " + datasetFolder + xfp + " doesn't exist")	
	
	print("--- " + arguments.algorithm.upper() + " done in %s seconds ---" % (time.time() - startTime))

	print("Split dataset into training and test set [0.80] [0.20]")
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

	
	print("Launching SVM...")
	startTime = time.time()

	if arguments.training == True:
		trainingTime = time.time()
		clf = svm.SVC()
		print("Start training...")
		clf.fit(x_train, y_train)
		joblib.dump(clf, 'model/svm.pkl') 
		print("--- Training done in %s seconds ---" % (time.time() - trainingTime))

	clf = joblib.load('model/svm.pkl') 
	print("Start testing...")
	predicted = clf.predict(x_test)

	print("--- SVM done in %s seconds ---" % (time.time() - startTime))

	print("Accuracy: " + str(accuracy_score(y_test, predicted)))
	
if __name__ == "__main__":
	main()



