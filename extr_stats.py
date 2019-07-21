import sys
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class Eye():
	def __init__(self, x_left, x_right):
		self.x_left = x_left
		self.x_right = x_right

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


'''
	Works fine, but not the best
'''
def computeStats(x_left, x_right):
	x_left = np.array(x_left)
	x_right = np.array(x_right)

	print("LEFT")
	print(stats.describe(x_left))
	print("RIGHT")
	print(stats.describe(x_right))


'''
	Works fine, but not the best
'''
def computeVel(x_left, x_right):
	print("\nVELOCITY")
	v_left = []
	v_right = []
	last_left = x_left[0]
	last_right = x_right[0]
	for i in range(1, len(x_left)):
		v_left.append(x_left[i] - last_left)
		v_right.append(x_right[i] - last_right)
		last_left = x_left[i]
		last_right = x_right[i]

	print("LEFT")
	print(stats.describe(v_left))
	print("RIGHT")
	print(stats.describe(v_right))

	return v_left, v_right

'''
	Works fine, but not the best
'''
def computeAccel(v_left, v_right):
	print("\nACCELERATION")
	a_left = []
	a_right = []
	last_left = v_left[0]
	last_right = v_right[0]
	for i in range(1, len(v_left)):
		a_left.append(v_left[i] - last_left)
		a_right.append(v_right[i] - last_right)
		last_left = v_left[i]
		last_right = v_right[i]

	print("LEFT")
	print(stats.describe(v_left))
	print("RIGHT")
	print(stats.describe(v_right))

'''
	This is essentially the only useful method here. We should be
		computing the velocity and acceleration using the numpy 
		gradient method
'''
def computeAndPlot(xVals, fileName):
	# compute x velocity and acceleration
	xpos = np.asarray(xVals)
	xVel = np.gradient(xpos)
	xAcc = np.gradient(xVel)

	# compute and print statistics on input data
	print("\nPosition")
	print(stats.describe(xpos))
	print("\nVelocity")
	print(stats.describe(xVel))
	print("\nAcceleration")
	print(stats.describe(xAcc))

	# plot pos,velocity, accel
	color = 'blue' if "intox" in fileName else 'r'
	rng = range(len(xpos))

	# plt.scatter(rng[:5500], xpos[:5500], c=color, alpha=0.5, label='')
	# plt.title("X Position")
	# axes = plt.gca()
	# axes.set_ylim([0,180])
	# plt.savefig(fileName + "_positionPlot")
	# plt.close()

	# plt.plot(rng[:6000], xVel[:6000], '-o', color=color, alpha=0.5, label='')
	# plt.title("X Velocity")
	# axes = plt.gca()
	# axes.set_ylim([-40,41])
	# plt.savefig(fileName + "_velocityPlot")
	# plt.close()

	# plt.hist(xAcc[:6000], normed=True, color = color, alpha=0.5, label='')
	plt.plot(rng[:6000], xAcc[:6000], '-o', color=color, alpha=0.5, label='')
	plt.title("X Acceleration")
	axes = plt.gca()
	axes.set_ylim([-25,30])
	plt.savefig(fileName + "_accelerationPLot")
	plt.close()



if __name__ == '__main__':

	vid1 = "dax_extracted/AC_1340__072_DD.txt"
	# vid2 = "dax_extracted/BB_00_RS_080918.txt"
	vid2 = "dax_extracted/AC_1380_000_DD.txt"


	print("SOBER VIDEO")
	x_l, x_r = readFiles(vid2)
	computeAndPlot(x_l, "soberLeftsameP_newAccel")
	# computeStats(x_l, x_r)
	# v_l, v_r = computeVel(x_l, x_r)
	# computeAccel(v_l, v_r)


	print("\n\nINTOXICATED VIDEO")
	x_l, x_r = readFiles(vid1)
	computeAndPlot(x_l, "intoxLeftsameP_newAccel")
	# computeStats(x_l, x_r)
	# v_l, v_r = computeVel(x_l, x_r)
	# computeAccel(v_l, v_r)




