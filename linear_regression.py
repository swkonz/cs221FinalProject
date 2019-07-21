#!/usr/bin/env python

import pickle
import numpy as np

left_features = pickle.load(open("leftFeatures.p", "rb"))
right_features = pickle.load(open("rightFeatures.p", "rb"))

def linearRegression(features):
	feat_len = len(features[0][0])
	weights = np.random.rand(feat_len)
	T = 10000
	eta = 0.00001

	def trainLoss(w):
		D_train = len(features)
		s = 0
		for feature in features:
			for i, feat in enumerate(feature[0]):
				s += 2 * (weights[i] * feat - feature[1]) * feature[1]
		return (1/float(D_train)) * s

	for t in range(T):
		gradient = trainLoss(weights)
		weights = weights - eta * gradient

	return weights

print linearRegression(left_features)