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
import shutil #remove folder with all files

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
	return numpy.matrix(imgObj) 

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
	raw_input("Hit enter to continue...")

# Convert an img array into Image object
def getImgObjFromArray(imgArray):
	return Image.fromarray(imgArray.astype('uint8'), 'L')

# Create the folder LBP inside the dataset folder, in order to save the LBP image
def createFolderLBP(dataset, algorithm):
	path = "datasets/" + algorithm + "/" + dataset +"/"
	if not os.path.exists(path):
		os.makedirs(path)
	else:
		shutil.rmtree(path, ignore_errors=True)
		os.makedirs(path)

# Create the folder LBP inside the dataset folder, in order to save the LBP image
def createFolder(dataset):
	path = "datasets/" + dataset +"/"
	if not os.path.exists(path):
		os.makedirs(path)
	else:
		shutil.rmtree(path, ignore_errors=True)
		os.makedirs(path)

# Passing an image, a dataset name and file name, store the image in png format
def saveImgLBP(img, dataset, filename, algorithm):
	path = "datasets/" + algorithm + "/" + dataset +"/"
	img.save(path + filename)

# Passing an image, a dataset name and file name, store the image in png format
def saveImgFromArray(img, dataset, filename):
	path = "datasets/" + "/" + dataset +"/"
	img.save(path + filename)


# Equalize the img histogram passed as a parameter
def histogramEqualization(imgObj):
	imgArray = getImgArray(imgObj)
	eqImg = cv2.equalizeHist(imgArray)
	return getImgObjFromArray(eqImg)


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

# Rotate an img
def rotateImg(imgArray, angle):	
	rows,cols = imgArray.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
	dst = cv2.warpAffine(imgArray,M,(cols,rows))
	return dst

# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html
def getHistogram(imgArray):
	hist, bin_edges = numpy.histogram(imgArray, density=True)
	return hist