# facerec.py
import cv2, sys, numpy, os, argparse
from utils.utils import *
from sklearn import svm


def main():

	parser = argparse.ArgumentParser(description='Run the local binary patterns algorithm using either a single process or multiple processes.')
	parser.add_argument('--algorithm', dest='algorithm', type=str, default='lbp', help='Algorithm to use: "lbp" or "fisherface"')

	arguments = parser.parse_args()

	# Security check about the algorithm
	if(arguments.algorithm != "lbp" and arguments.algorithm != "fisherface"):
		print("Algorithm not valid. Choose between lbp or fisherface")
		return

	size = 4
	haar_file = 'haarcascade_frontalface_default.xml'
	datasets = 'datasets'
	# Part 1: Create fisherRecognizer

	# Create a list of images and a list of corresponding names
	(images, lables, names, id) = getDatasets(datasets)
	# Size of images
	(width, height) = (130, 100)

	# Create a Numpy array from the two lists above
	(images, lables) = [numpy.array(lis) for lis in [images, lables]]

	print('Training dataset using ' + arguments.algorithm)
	if arguments.algorithm == "fisherface":
		# OpenCV trains a model from the images
		model = cv2.face.createFisherFaceRecognizer()
		model.train(images, lables)
	if arguments.algorithm == "lbp":
		x = [];
		
		for img in images: 
			lbp = LBP(img)
			# Concatenate the various histogram, the resulting histogram is append into feature vector
			x.append(lbp)

		model = svm.LinearSVC()
		model.fit(x, lables)
	print('Training dataset using ' + arguments.algorithm + ' done')


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

			if arguments.algorithm == "fisherface":
				# Try to recognize the face using fisherface
				prediction = model.predict(face_resize) #fisherface get the label as 0
				#print("[Debug]: " + names[prediction] + "'s face found")
			if arguments.algorithm == "lbp":
				lbp = [];
				lbp.append(LBP(face_resize))
				prediction = model.predict(lbp)
				prediction = prediction[0] # SVM get and array with the class [0]
				#print("[Debug]: " + names[prediction] + "'s face found")

			cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

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
