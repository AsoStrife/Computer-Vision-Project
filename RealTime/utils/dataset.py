'''
@author Andrea Corriga
@contact me@andreacorriga.com
@date 2018
@version 1.0
'''

import os

'''
 Starting by a dataset name, this function return 
 classes => an array with all classes (subfolders inside dataset folder)
 filename => an array with all images filenames
 xFilepath => an array with all images name with path /dataset/classes_folder/filename.pgm
 y => an array with the relative label of filename||xFilepath
'''
def getDataset(dataset):
	directory = os.getcwd() + "/datasets/" + dataset +"/"

	classes = []
	filename = []
	xFilepath = []
	y = []

	for root, dirs, files in os.walk(directory):
		for dir in dirs:
			classes.append(dir)

	for imgClass in classes:
		for file in os.listdir(directory + imgClass):
			y.append(imgClass)
			xFilepath.append(imgClass + "/" + file)
			filename.append(file)

	return classes, filename, xFilepath, y


def getPersonalDataset():
	directory = os.getcwd() + "/datasets/personal/"

	x = []

	for root, dirs, files in os.walk(directory):
		for file in files:
			x.append("datasets/personal/" + file)

	return x