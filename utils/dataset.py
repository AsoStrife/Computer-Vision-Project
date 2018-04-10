'''
@author Andrea Corriga
@contact me@andreacorriga.com
@date 2018
@version 1.0
'''

import os

def getDataset(dataset):
	directory = os.getcwd() + "/datasets/" + dataset +"/"

	classes = []
	xFilepath = []
	y = []

	for root, dirs, files in os.walk(directory):
		for dir in dirs:
			classes.append(dir)

	for imgClass in classes:
		for file in os.listdir(directory + imgClass):
			y.append(imgClass)
			xFilepath.append(imgClass + "/" + file)


	return classes, xFilepath, y