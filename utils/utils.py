'''
@author Andrea Corriga
@contact me@andreacorriga.com
@date 2018
@version 1.0
'''

import os.path
import numpy
import dlib
import cv2
from PIL import Image

# Other method to read an image starting by filepath
#img = cv2.imread(filepath)

# Read an image starting by the file path and and return the Image object converted in grayscale
def imgRead(filepath):
	if os.path.isfile(filepath):
		return Image.open(filepath).convert("L")
	else:
		return

# Return the size of an image
def getImgSize(imgObj):
	return imgObj.size

# Passing an Image Object return the equivalent numpy array values
def getImgArray(imgObj):
	return numpy.array(imgObj) 

# Check if an Image contain one or more valid faces. 
def faceDetect(imgArray):
	detector = dlib.get_frontal_face_detector()
	# Run the face detector, upsampling the image 1 time to find smaller faces.
	dets = detector(imgArray, 1)
	
	if len(dets) >= 1:
		return True
	else:
		return False
	
	'''DEBUG SHOW FACE RETTANGLE
	print "number of faces detected: ", len(dets)
	win = dlib.image_window()
	win.set_image(img)
	win.add_overlay(dets)
	raw_input("Hit enter to continue")
	'''

# Usefull to debug, show in a new Windows the image passed as a parameters
def imgShow(imgArray):
	win = dlib.image_window()
	win.set_image(imgArray)
	raw_input("Hit enter to continue")

# Convert an img array into Image object
def getImgObjFromArray(imgArray):
	return Image.fromarray(imgArray.astype('uint8'), 'L')

# Passing an image, a dataset name and file name, store the image in png format
def saveImgFromArray(imgArray, dataset, filename):
	img = getImgObjFromArray(imgArray)
	save("datasets/" + dataset + "/LBP/" + filename +".png")

# Equalize the img histogram passed as a parameter
def histogramEqualization(imgObj):
	imgArray = getImgArray(imgObj)
	eqImg = cv2.equalizeHist(imgArray)
	return getImgObjFromArray(eqImg)