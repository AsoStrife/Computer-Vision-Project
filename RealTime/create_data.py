#creating database
import cv2, sys, numpy, os, argparse

def main():
	parser = argparse.ArgumentParser(description='Run the local binary patterns algorithm using either a single process or multiple processes.')
	parser.add_argument('--name', dest='name', type=str, default="", help='The name of the person you want to store the face')

	arguments = parser.parse_args()

	if(arguments.name == ""):
		print("Name can't be null. Please insert a name")
		return

	haar_file = 'haarcascade_frontalface_default.xml'

	#All the faces data will be present this folder
	datasets = 'datasets'  
	#These are sub data sets of folder, for my faces I've used my name
	sub_data = arguments.name     

	# If not exist the datasets folder
	if not os.path.isdir(datasets):
		os.mkdir(datasets)

	# Create the name folder
	path = os.path.join(datasets, sub_data)
	if not os.path.isdir(path):
		os.mkdir(path)

	# defining the size of images 
	(width, height) = (130, 100)    


	face_cascade = cv2.CascadeClassifier(haar_file)
	webcam = cv2.VideoCapture(0) 

	# The program loops until it has 30 images of the face.
	count = 1

	# Keep open the webcam until click ESC
	while True:
		(_, im) = webcam.read()
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 4)
		
		# Store 30 pic of the face
		for (x,y,w,h) in faces:
			cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
			face = gray[y:y + h, x:x + w]
			if count < 31:
				face_resize = cv2.resize(face, (width, height))
				cv2.imwrite('%s/%s.png' % (path,count), face_resize)
			count += 1
		
		cv2.imshow('OpenCV', im)
		key = cv2.waitKey(10)
		if key == 27:
			break

if __name__ == "__main__":
	main()