import pickle
import matplotlib.pyplot as plt
import numpy as np

# reads in data from pickle file (exact format it was saved as is returned)
def readFeaturesFromPickle(fileName):
	return pickle.load( open(fileName, "rb") )

if __name__ == '__main__':

	# read in feature vectors
	left_features = readFeaturesFromPickle("leftFeatures.p")
	right_features = readFeaturesFromPickle("rightFeatures.p")

	left_features_update = readFeaturesFromPickle("leftFeatures_update.p")
	right_features_update = readFeaturesFromPickle("rightFeatures_update.p")

	# try plotting num elems above threshold
	# x_sober = []
	# y_sober = []
	# x_drunk = []
	# y_drunk = []
	# for i in range(len(left_features_update)):
	# 	c = left_features_update[i][1]
	# 	if (c == 0) :
	# 		x_sober.append(left_features_update[i][0][3])
	# 		y_sober.append(left_features_update[i][0][4])
	# 		x_sober.append(right_features_update[i][0][3])
	# 		y_sober.append(right_features_update[i][0][4])
	# 	else:
	# 		x_drunk.append(left_features_update[i][0][3])
	# 		y_drunk.append(left_features_update[i][0][4])
	# 		x_drunk.append(right_features_update[i][0][3])
	# 		y_drunk.append(right_features_update[i][0][4])

	# # plot it
	# l = plt.scatter(x_sober, y_sober, alpha=0.5, label='sober')
	# plt.title("Number of elems over threshold")
	# plt.xlabel("Number velocity elems over threshold")
	# plt.ylabel("Number of acceleration elems over threshold")
	# # axes = plt.gca()
	# # axes.set_ylim([0,180])
	# r = plt.scatter(x_drunk, y_drunk, alpha=0.5, label='drunk')
	# plt.legend(handles=[l, r])
	# plt.savefig("num_elems_over_thresh")
	# plt.close()

	# try plotting num SF by avg velocity
	# x_sober = []
	# y_sober = []

	# x_drunk = []
	# y_drunk = []
	# for i in range(len(left_features_update)):
	# 	c = left_features_update[i][1]
	# 	if (c == 0) :
	# 		x_sober.append(left_features[i][0][0])
	# 		y_sober.append(left_features[i][0][1])
	# 		x_sober.append(right_features[i][0][0])
	# 		y_sober.append(right_features[i][0][1])
	# 	else:
	# 		x_drunk.append(left_features[i][0][0])
	# 		y_drunk.append(left_features[i][0][1])
	# 		x_drunk.append(right_features[i][0][0])
	# 		y_drunk.append(right_features[i][0][1])

	# # plot it
	# l = plt.scatter(x_sober, y_sober, alpha=0.5, label='sober')
	# plt.title("Number of Sign Flips by X velocity")
	# plt.xlabel("Number of Sign Flips")
	# plt.ylabel("Mean dist of SF from center")
	# # axes = plt.gca()
	# # axes.set_ylim([0,180])
	# r = plt.scatter(x_drunk, y_drunk, alpha=0.5, label='drunk')
	# plt.legend(handles=[l, r])
	# plt.savefig("num_SF_plt")
	# plt.close()

	# try plotting mean x velocity
	# x_sober = []
	# y_sober = []

	# x_drunk = []
	# y_drunk = []
	# for i in range(len(left_features_update)):
	# 	c = left_features_update[i][1]
	# 	if (c == 0) :
	# 		x_sober.append(left_features_update[i][0][5])
	# 		y_sober.append(left_features_update[i][0][6])
	# 		x_sober.append(right_features_update[i][0][5])
	# 		y_sober.append(right_features_update[i][0][6])
	# 	else:
	# 		x_drunk.append(left_features_update[i][0][5])
	# 		y_drunk.append(left_features_update[i][0][6])
	# 		x_drunk.append(right_features_update[i][0][5])
	# 		y_drunk.append(right_features_update[i][0][6])

	# # plot it
	# l = plt.scatter(x_sober, y_sober, alpha=0.5, label='sober')
	# plt.title("Mean X velocity vs Mean X acceleration")
	# plt.xlabel("Mean X velocity")
	# plt.ylabel("Mean X acceleration")
	# # axes = plt.gca()
	# # axes.set_ylim([0,180])
	# r = plt.scatter(x_drunk, y_drunk, alpha=0.5, label='drunk')
	# plt.legend(handles=[l, r])
	# plt.savefig("mean_value_plot")
	# plt.close()

	# plotting Vel/Acc variance
	x_sober = []
	y_sober = []

	x_drunk = []
	y_drunk = []
	for i in range(len(left_features_update)):
		c = left_features_update[i][1]
		if (c == 0) :
			x_sober.append(left_features_update[i][0][7])
			y_sober.append(left_features_update[i][0][8])
			x_sober.append(right_features_update[i][0][7])
			y_sober.append(right_features_update[i][0][8])
		else:
			x_drunk.append(left_features_update[i][0][7])
			y_drunk.append(left_features_update[i][0][8])
			x_drunk.append(right_features_update[i][0][7])
			y_drunk.append(right_features_update[i][0][8])

	# plot it
	l = plt.scatter(x_sober, y_sober, alpha=0.5, label='sober')
	plt.title("Variance X velocity vs Variance X acceleration")
	plt.xlabel("Variance X velocity")
	plt.ylabel("Variance X acceleration")
	# axes = plt.gca()
	# axes.set_ylim([0,180])
	r = plt.scatter(x_drunk, y_drunk, alpha=0.5, label='drunk')
	plt.legend(handles=[l, r])
	plt.savefig("variance_value_plot")
	plt.close()

