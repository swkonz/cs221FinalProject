import os
import numpy as np
import pickle
from scipy.stats import skew


'''
comput features from pupil point files
compute feature vectors for each eye individually
feature vectors are as follows
 - Number of sign flips for x velocity
 - avg distance from center position when sign flip occurs
 - avg X velocity
 - avg X acceleration
 - Varaince in X velocity
 - Variance in X Acceleration
'''
def extractFeatures(dirPath):
	files = os.listdir(dirPath)

	left_features = []
	right_features = []

	# loop through all text files with pupil points
	for file in files:
		file_path = os.path.join(dirPath, file)

		if(not os.path.isfile(file_path)):
			print("Not a file: " + file_path)
			continue
		elif(not file.split(".")[1] == "txt"):
			print("Not a text file: " + file_path)
			continue

		cur_left, cur_right = readFiles(file_path)

		# generate classification value: 0 if sober, 1 else
		classification = 0 if "_00" in file else 1

		# left_features.append(computeVector(cur_left, classification))
		# right_features.append(computeVector(cur_right, classification))

		left_features.append(computeVector_update(cur_left, classification))
		right_features.append(computeVector_update(cur_right, classification))

	# save generated feature vectors to pickle files for easy loading
	print("Writing to pickle file...")
	pickle.dump( left_features, open("leftFeatures_update.p", "wb") )
	pickle.dump( right_features, open("rightFeatures_update.p", "wb") )
	print("Done.")

def computeVector_update(rawValues, classification):

	xPos = np.asarray(rawValues)
	xVel = np.gradient(xPos)
	xAcc = np.gradient(xVel)

	# compute means
	xPos_mean = np.mean(xPos)
	xVel_mean = np.mean(xVel)
	xAcc_mean = np.mean(xAcc)

	# compute variance
	xVel_var = np.var(xVel)
	xAcc_var = np.var(xAcc)

	#compute skewness
	xVel_skew = skew(xVel)
	xAcc_skew = skew(xAcc)

	# compute number of sign flips and dist from center for sign flip
	last_sign = np.sign(xVel[0])
	num_signFlips = 0
	num_vel_elems_over_threshold = 0
	num_acc_elems_over_threshold = 0
	velocity_threshold = 30
	acceleration_threshold = 20
	for i in range(1, len(xVel)):
		# value thresholding
		if (np.abs(xVel[i]) > velocity_threshold):
			num_vel_elems_over_threshold += 1

		if (np.abs(xAcc[i]) > acceleration_threshold):
			num_acc_elems_over_threshold += 1

		# sign flip
		cur_sign = np.sign(xVel[i])
		if(last_sign != cur_sign and last_sign != 0 and cur_sign != 0):
			num_signFlips += 1
		last_sign = cur_sign

	# create full feature vector and put into list
	return ([num_signFlips, xVel_skew, xAcc_skew,
							num_vel_elems_over_threshold, 
							num_acc_elems_over_threshold,
							xVel_mean, xAcc_mean, 
							xVel_var, xAcc_var],
							classification)

# Actually compute a given feature vector from a list of raw x,y coords
def computeVector(rawValues, classification):

	xPos = np.asarray(rawValues)
	xVel = np.gradient(xPos)
	xAcc = np.gradient(xVel)

	# compute means
	xPos_mean = np.mean(xPos)
	xVel_mean = np.mean(xVel)
	xAcc_mean = np.mean(xAcc)

	# compute variance
	xVel_var = np.var(xVel)
	xAcc_var = np.var(xAcc)

	# compute number of sign flips and dist from center for sign flip
	last_sign = np.sign(xVel[0])
	num_signFlips = 0
	avg_dist_from_center_SF = 0
	for i in range(1, len(xVel)):
		cur_sign = np.sign(xVel[i])
		if(last_sign != cur_sign and last_sign != 0 and cur_sign != 0):
			num_signFlips += 1
			avg_dist_from_center_SF += xPos[i]
		last_sign = cur_sign

	avg_dist_from_center_SF /= num_signFlips

	# create full feature vector and put into list
	return ([num_signFlips, avg_dist_from_center_SF, 
							xVel_mean, xAcc_mean, 
							xVel_var, xAcc_var],
							classification)


'''
Accepts a parameter pointPath with the path to the text files 
	with the extracted pupil points
	@param pointPath: file path to feature text file
	@return 2D array of all file points
'''
def readFiles(fileToCompute):

	pointFile = open(fileToCompute)

	x_left = []
	x_right = []
	for line in pointFile:
		line_split = line.split(" ")
		x_left.append(int(line_split[0]))
		x_right.append(int(line_split[2]))

	return x_left, x_right



if __name__ == '__main__':
	extractFeatures("./dax_extracted/")



