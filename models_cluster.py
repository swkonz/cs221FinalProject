import numpy as np
from scipy.cluster.vq import kmeans2, whiten
import pickle

# for KNN
# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# for reading in feature vectors from text files
def readFeatureVectors(soberFileName, drunkFileName):
	soberFile = open(soberFileName)
	data = []
	for line in soberFile:
		line_split = line.split(",")
		vec = []
		for elem in line_split:
			vec.append(float(elem))
		data.append((vec, 0))

	drunkFile = open(drunkFileName)
	for line in drunkFile:
		line_split = line.split(",")
		vec = []
		for elem in line_split:
			vec.append(float(elem))
		data.append((vec, 1))

	return data

# reads in data from pickle file (exact format it was saved as is returned)
def readFeaturesFromPickle(fileName):
	return pickle.load( open(fileName, "rb") )

# split data when all data is in a single data array
def splitData(data):
	trainingSet = []
	testSet = []
	splitRatio = 0.2
	n = len(data)
	data_np = np.array(data)
	np.random.shuffle(data_np)
	index = int(n * (1-splitRatio))
	trainingSet, testSet = data_np[:index], data_np[index:]
	return trainingSet, testSet, data_np

# run KMeans on a given data cluster and print results
def handle_Kmeans(data_full):

	# split X and Y from input data
	x = data_full[:, 0]
	y = data_full[:, 1]

	x_whitened = whiten(x.tolist())
	
	# run Kmeans
	centroids, labels = kmeans2(x_whitened, 2, iter=100)

	# evaluate results
	print("Done classifying")
	print(confusion_matrix(y.tolist(), labels))
	print(classification_report(y.tolist(), labels))

	return accuracy_score(y.tolist(), labels, normalize=True)


# run K-Nearest Neighbor on given data cluster and print results
def handle_KNN(trainingSet, testingSet, norm=False):

	# split X and Y from input data
	x_train = trainingSet[:, 0].tolist()
	y_train = trainingSet[:, 1].tolist()

	x_test = testingSet[:, 0].tolist()
	y_test = testingSet[:, 1].tolist()

	if (norm == True):
		# setup the data normalization
		scaler = StandardScaler()
		scaler.fit(x_train)
		x_train = scaler.transform(x_train)
		x_test = scaler.transform(x_test)

	# train the actual classifier
	classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
	classifier.fit(x_train, y_train)

	# run on the test data
	y_pred = classifier.predict(x_test)

	print("Done classifying")
	print(confusion_matrix(y_test, y_pred))
	print(classification_report(y_test, y_pred))

	# return accuracy score
	return accuracy_score(y_test, y_pred, normalize=True)


if __name__ == '__main__':

	############################################################
	# Using original Feature Vectors
	############################################################

	# for reading in data from txt feature files
	# data_full = readFeatureVectors("sober.txt", "drunk.txt")
	# trainingSet, testSet, data = splitData(data_full)

	# read in feature vectors from pickle files
	# left_features = readFeaturesFromPickle("leftFeatures.p")
	# right_features = readFeaturesFromPickle("rightFeatures.p")

	############################################################
	# K Means Testing
	############################################################
	'''
	This is just running on the raw feature inputs with an unbalanced
	dataset. This gives an unrealistic demonstration. It just seems
	to place all datapoints into a single cluster. With whitening, 
	the clusters become slightly more defined.
	'''
	# left features
	# handle_Kmeans(np.array(left_features))

	# try running on right features instead
	# handle_Kmeans(np.array(right_features))
	'''
	The spread for running KMeans on the right features is better.
	This is probably due to the fact that the pupil localization is
	more accurate on the right pupil since it seems that the right
	pupil is closer to the camera, and thus the computed pupil
	location is more consistent
	'''

	# run kmeans many time with multiple random initializations
	# n = 250
	# score_right = score_left = 0
	# for i in range(n):
	# 	score_right += handle_Kmeans(np.array(right_features))
	# 	score_left += handle_Kmeans(np.array(left_features))

	# # compute total score 
	# total_left = score_left / float(n)
	# total_right = score_right / float(n)
	# print("accuracy of left: " + str(total_left))
	# print("accuracy of right: " + str(total_right))

	'''
	Output accuracy: Left: 50% Right: 50%
		No better than a coin flip. Looking at the distribution, it 
		seems like at any given random initialization it would 
		almost always place all data into a single category which 
		changed based on the randomly initialized centroids
	'''

	############################################################
	# K-Nearest Neighbors Testing
	############################################################

	# split into train and test sets - output is np array (not internals)
	# leftTrainingSet, leftTestSet, _ = splitData(left_features)
	# rightTrainingSet, rightTestSet, _ = splitData(right_features)

	# use the same dataset to run KNN multiple times to get accuracy
	# n = 250
	# score_right = score_left = 0
	# for i in range(n):
	# 	score_right += handle_KNN(rightTrainingSet, rightTestSet, norm=True)
	# 	score_left += handle_KNN(leftTrainingSet, leftTestSet, norm=True)

	# # compute total score 
	# total_left = score_left / float(n)
	# total_right = score_right / float(n)
	# print("accuracy of left: " + str(total_left))
	# print("accuracy of right: " + str(total_right))

	'''
	Overall Accuracy: 71% accuracy for left eye classification
					  60% accuracy for right eye classification
	'''
	
	# running KNN on left
	# handle_KNN(leftTrainingSet, leftTestSet, norm=True)

	# run KNN on right 
	# handle_KNN(rightTrainingSet, rightTestSet, norm=True)



	############################################################
	# Using UPDATED Feature Vectors
	############################################################
	
	# for reading in data from txt feature files
	# data_full = readFeatureVectors("sober.txt", "drunk.txt")
	# trainingSet, testSet, data = splitData(data_full)

	# read in feature vectors from pickle files
	left_features_update = readFeaturesFromPickle("leftFeatures_update.p")
	right_features_update = readFeaturesFromPickle("rightFeatures_update.p")

	############################################################
	# K Means Testing
	############################################################

	# run kmeans many time with multiple random initializations
	# n = 250
	# score_right = score_left = 0
	# for i in range(n):
	# 	score_right += handle_Kmeans(np.array(right_features_update))
	# 	score_left += handle_Kmeans(np.array(left_features_update))

	# # compute total score 
	# total_left = score_left / float(n)
	# total_right = score_right / float(n)
	# print("accuracy of left: " + str(total_left))
	# print("accuracy of right: " + str(total_right))

	'''
	Output accuracy: Left:52%  Right: 51%
		a weirdly small increase in accuracy, but it was consistent 
		across multiple random initializations
	'''

	############################################################
	# K-Nearest Neighbors Testing
	############################################################

	# split into train and test sets - output is np array (not internals)
	leftTrainingSet, leftTestSet, _ = splitData(left_features_update)
	rightTrainingSet, rightTestSet, _ = splitData(right_features_update)

	# use the same dataset to run KNN multiple times to get accuracy
	n = 250
	score_right = score_left = 0
	for i in range(n):
		score_right += handle_KNN(rightTrainingSet, rightTestSet, norm=True)
		score_left += handle_KNN(leftTrainingSet, leftTestSet, norm=True)

	# compute total score 
	total_left = score_left / float(n)
	total_right = score_right / float(n)
	print("accuracy of left: " + str(total_left))
	print("accuracy of right: " + str(total_right))

	'''
	Overall Accuracy: 74% accuracy for left eye classification
					  79% accuracy for right eye classification
	'''
	
	# running KNN on left
	# handle_KNN(leftTrainingSet, leftTestSet, norm=True)

	# run KNN on right 
	# handle_KNN(rightTrainingSet, rightTestSet, norm=True)






