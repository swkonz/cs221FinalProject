import pickle
import numpy as np

# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

def handle_logisticRegression(trainingSet, testSet):
	
	# split X and Y from input data
	x_train = trainingSet[:, 0].tolist()
	y_train = trainingSet[:, 1].tolist()

	x_test = testSet[:, 0].tolist()
	y_test = testSet[:, 1].tolist()

	# actual regression classifier
	LR = LogisticRegression(penalty='l2', class_weight='balanced', solver='liblinear' , max_iter=1000, fit_intercept=True)
	LR.fit(x_train, y_train)

	y_pred = LR.predict(x_test)
	y_pred_proba = LR.predict_proba(x_test)

	# evaluate results
	# print("Done classifying")

	# for i in range(len(y_pred)):
	# 	if(y_pred[i] != y_test[i]):
	# 		print("Misclassified " + str(y_test[i]) + " as " + str(y_pred[i]))
	# 		print(y_pred_proba[i])

	# print(confusion_matrix(y_test, y_pred))
	# print(classification_report(y_test, y_pred))

	return accuracy_score(y_test, y_pred, normalize=True)


if __name__ == '__main__':

	# read in feature vectors from pickle files
	# left_features = readFeaturesFromPickle("leftFeatures.p")
	# right_features = readFeaturesFromPickle("rightFeatures.p")

	##############################################
	# Logistic Regression 
	##############################################
	# leftTrainingSet, leftTestSet, _ = splitData(left_features)
	# rightTrainingSet, rightTestSet, _ = splitData(right_features)

	# run on left eyes
	# handle_logisticRegression(leftTrainingSet, leftTestSet)

	# run on left eyes
	# handle_logisticRegression(rightTrainingSet, rightTestSet)

	# run many iterations to get actual accuracy score
	# n = 250
	# score_right = score_left = 0
	# for i in range(n):
	# 	score_right += handle_logisticRegression(rightTrainingSet, rightTestSet)
	# 	score_left += handle_logisticRegression(leftTrainingSet, leftTestSet)

	# # compute total score 
	# total_left = score_left / float(n)
	# total_right = score_right / float(n)
	# print("accuracy of left: " + str(total_left))
	# print("accuracy of right: " + str(total_right))

	'''
	Accuracy:
		With L2 Loss: 61% Left
					  71% right
		With L1 Loss: 
					  58% left
					  66% right
		With L2 Loss and Intercept scaling for Imbalanced data:
					  71% Left
					  73% right
		With L1 Loss and Intercept scaling for Imbalanced data:
					  66% Left
					  63% right
	'''

	#######################################################
	# Using UPDATED Feature Vectors
	#######################################################

	# read in feature vectors from pickle files
	left_features_update = readFeaturesFromPickle("leftFeatures_update.p")
	right_features_update = readFeaturesFromPickle("rightFeatures_update.p")

	##############################################
	# Logistic Regression 
	##############################################
	leftTrainingSet, leftTestSet, _ = splitData(left_features_update)
	rightTrainingSet, rightTestSet, _ = splitData(right_features_update)

	# run on left eyes
	# handle_logisticRegression(leftTrainingSet, leftTestSet)

	# run on left eyes
	# handle_logisticRegression(rightTrainingSet, rightTestSet)

	# run many iterations to get actual accuracy score
	n = 250
	score_right = score_left = 0
	for i in range(n):
		score_right += handle_logisticRegression(rightTrainingSet, rightTestSet)
		score_left += handle_logisticRegression(leftTrainingSet, leftTestSet)

	# compute total score 
	total_left = score_left / float(n)
	total_right = score_right / float(n)
	print("accuracy of left: " + str(total_left))
	print("accuracy of right: " + str(total_right))

	'''
	Accuracy:
		With L2 Loss: 58% Left
					  61% right
		With L1 Loss: 
					  61% left
					  55% right
		With L2 Loss and Intercept scaling for Imbalanced data:
					  63% Left
					  66% right
		With L1 Loss and Intercept scaling for Imbalanced data:
					  58% Left
					  68% right

	No improvement in Logistic regression with updated feature 
	vectors. Somewhat of a surprise.
	'''

