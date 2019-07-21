import os
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

def load_data(dirPath):
	files = os.listdir(dirPath)

	left_data = []
	right_data = []

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

		left_data.append( (cur_left, classification) )
		right_data.append( (cur_right, classification) )

	return left_data, right_data


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
	return trainingSet[:, 0], trainingSet[:, 1], testSet[:, 0], testSet[:, 1]


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

def runLSTM(X_train, y_train, X_test, y_test):
	max_seq_len = 6000
	embedding_vecor_length = 124

	# setup the model
	model = Sequential()
	model.add(Embedding(max_seq_len, embedding_vecor_length, input_length=max_seq_len))
	model.add(LSTM(124))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	print(model.summary())
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)

	# evaluate performance
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

if __name__ == '__main__':
	left, right = load_data("./dax_extracted/")

	# ignore the right eye data for now and just use the left
	X_train, y_train, X_test, y_test = splitData(left)

	# truncate and pad sequences
	max_seq_len = 6000
	X_train = sequence.pad_sequences(X_train, maxlen=max_seq_len)
	X_test = sequence.pad_sequences(X_test, maxlen=max_seq_len)

	# run the LSTM model on the input data
	runLSTM(X_train, y_train, X_test, y_test)

