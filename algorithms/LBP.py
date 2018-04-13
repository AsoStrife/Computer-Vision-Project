'''
@author Andrea Corriga
@contact me@andreacorriga.com
@date 2018
@version 1.0
'''

from PIL import Image
import numpy

class LBP:
	def __init__(self, img):
		# Convert the image to grayscale
		self.image = img
		# Size of original image
		self.width = self.image.size[0]
		self.height = self.image.size[1]
		# Future value of LBP image
		self.patterns = []


	# Starting by the image, calculate the LBP values and store into self.patterns variable
	def execute(self):
		pixels = list(self.image.getdata())
		pixels = [pixels[i * self.width:(i + 1) * self.width] for i in xrange(self.height)]

		# Calculate LBP for each non-edge pixel
		for i in xrange(1, self.height - 1):
			# Cache only the rows we need (within the neighborhood)
			previous_row = pixels[i - 1]
			current_row = pixels[i]
			next_row = pixels[i + 1]

			for j in xrange(1, self.width - 1):
				# Compare this pixel to its neighbors, starting at the top-left pixel and moving
				# clockwise, and use bit operations to efficiently update the feature vector
				pixel = current_row[j]
				pattern = 0
				pattern = pattern | (1 << 0) if pixel < previous_row[j-1] else pattern
				pattern = pattern | (1 << 1) if pixel < previous_row[j] else pattern
				pattern = pattern | (1 << 2) if pixel < previous_row[j+1] else pattern
				pattern = pattern | (1 << 3) if pixel < current_row[j+1] else pattern
				pattern = pattern | (1 << 4) if pixel < next_row[j+1] else pattern
				pattern = pattern | (1 << 5) if pixel < next_row[j] else pattern
				pattern = pattern | (1 << 6) if pixel < next_row[j-1] else pattern
				pattern = pattern | (1 << 7) if pixel < current_row[j-1] else pattern

				self.patterns.append(pattern)

	# This method return the LBP image generated (Image object)
	def getImage(self):
		result_image = Image.new(self.image.mode, (self.width - 2, self.height - 2))
		result_image.putdata(self.patterns)
		return result_image

	# Return the LBP patters as array
	def getImageArray(self): 
		return self.patterns

	# Return the histogram of the image
	def getHistogram(self):
		hist, bin_edges = numpy.histogram(self.patterns, density=True)
		return hist
