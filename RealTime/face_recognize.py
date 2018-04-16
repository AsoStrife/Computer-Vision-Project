# facerec.py
import cv2, sys, numpy, os, argparse
from skimage.feature import local_binary_pattern
from utils.utils import *
from sklearn import svm

def LBP(img): 
	lbp_value = local_binary_pattern(img, 8, 1)

	# Split img into 10*10 blocks
	shaped = blockshaped(lbp_value, 10, 13)

	# Calculate the histogram for each block
	xBlocks = []
	for s in shaped:
		xBlocks.append(getHistogram(s))

	return numpy.concatenate(xBlocks)

def main():

	parser = argparse.ArgumentParser(description='Run the local binary patterns algorithm using either a single process or multiple processes.')
	parser.add_argument('--algorithm', dest='algorithm', type=str, default='lbp', help='Algorithm to use: "lbp" or "elbp"')

	arguments = parser.parse_args()

	# Security check about the algorithm
	if(arguments.algorithm != "lbp" and arguments.algorithm != "fisherface"):
		print("Algorithm not valid. Choose between lbp or fisherface")
		return

	size = 4
	haar_file = 'haarcascade_frontalface_default.xml'
	datasets = 'datasets'
	# Part 1: Create fisherRecognizer
	print('Training...')
	# Create a list of images and a list of corresponding names
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
	(width, height) = (130, 100)

	# Create a Numpy array from the two lists above
	(images, lables) = [numpy.array(lis) for lis in [images, lables]]

	# OpenCV trains a model from the images
	# NOTE FOR OpenCV2: remove '.face'
	model = cv2.face.createFisherFaceRecognizer()
	model.train(images, lables)

	x = [];

	############ training with svm ################
	for img in images: 
		lbp = LBP(img)
		# Concatenate the various histogram, the resulting histogram is append into feature vector
		x.append(lbp)

	clf = svm.LinearSVC()
	print("Start training...")
	clf.fit(x, lables)
	############ training with svm ################


	# Part 2: Use fisherRecognizer on camera stream
	face_cascade = cv2.CascadeClassifier(haar_file)
	webcam  = cv2.VideoCapture(0)

	while True:
		(_, im) = webcam.read()
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		for (x,y,w,h) in faces:
			cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
			face = gray[y:y + h, x:x + w]
			face_resize = cv2.resize(face, (width, height))
			# Try to recognize the face
			prediction = model.predict(face_resize)
			cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

			lbp = [];
			lbp.append(LBP(face_resize))
			predict = clf.predict(lbp)
			print(prediction)

			if prediction < 500:
			   cv2.putText(im,'%s - %.0f' % (names[prediction],prediction),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
			else:
			  cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
			

		cv2.imshow('OpenCV', im)
		key = cv2.waitKey(10)
		if key == 27:
			break

if __name__ == "__main__":
	main()
