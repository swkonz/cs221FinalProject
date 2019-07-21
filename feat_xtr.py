#!/usr/bin/env python
import glob
import numpy as np

class Eye():
	def __init__(self, p, pp=None): # pp is short for Previous Point
		self.P = p # position
		self.V = (0 if not pp else p - pp.P) # velocity
		self.A = (0 if not pp else abs(self.V - pp.V)) # acceleration
		if pp:
			self.sf = (np.sign(self.V) != np.sign(pp.V) and np.sign(self.V) != 0 and np.sign(pp.V) != 0) # true (1) if flipped

class Frame():
	def __init__(self, vals, pf=None): # pf is short for Previous Frame
		self.left = Eye(vals[0], (None if not pf else pf.left))  # left eye
		self.right = Eye(vals[2], (None if not pf else pf.right)) # right eye

def extractFeature(dax_file_name):
	with open(dax_file_name) as dax_file:
		sum_positions = [0,0]
		num_frames = 0
		frames = [[int(c) for c in line.split(' ')] for line in dax_file.readlines()]
	
	if len(frames) == 0:
		return []
	prev_frame = Frame(frames[0])
	for vals in frames[1:]:
		new_frame = Frame(vals, prev_frame)
		num_frames += 1
		sum_positions[0] += new_frame.left.P
		sum_positions[1] += new_frame.right.P

	avg_positions = [float(sp)/num_frames for sp in sum_positions]

	num_sign_flips = [0,0] # implicit keys of these vectors: LX, RX 
	sum_velocities = [0,0]  # for getting the average velocity
	sum_accelerations = [0,0] # for getting the average acceleration
	sum_distances_from_centers = [0,0] # for getting avg distance from center

	prev_frame = Frame(frames[0])
	for vals in frames[1:]:
		new_frame = Frame(vals, prev_frame)

		num_sign_flips[0] += int(new_frame.left.sf)
		num_sign_flips[1] += int(new_frame.right.sf)

		sum_distances_from_centers[0] += (abs(new_frame.left.P - avg_positions[0]) if int(new_frame.left.sf)  else 0)
		sum_distances_from_centers[1] += (abs(new_frame.left.P - avg_positions[1]) if int(new_frame.right.sf) else 0)

		sum_velocities[0] += new_frame.left.V
		sum_velocities[1] += new_frame.right.V

		sum_accelerations[0] += new_frame.left.A
		sum_accelerations[1] += new_frame.right.A

		prev_frame = new_frame

	avg_velocities = [float(sv)/num_frames for sv in sum_velocities]
	avg_accelerations = [float(sa)/num_frames for sa in sum_accelerations]
	avg_distance_from_centers = [float(sd)/num_sign_flips[i] for i, sd in enumerate(sum_distances_from_centers)]

	return num_sign_flips + avg_velocities + avg_accelerations + avg_distance_from_centers

def runFeatureExtraction(dir_path):
	txt_file_names = glob.glob(dir_path)
	sober = ''
	drunk = ''

	for txt_file_name in txt_file_names:
		feature = extractFeature(txt_file_name)
		if '_00' in txt_file_name:
			sober += str(feature) + '\n'
		else:
			drunk += str(feature) + '\n'

	s_file = open('sober.txt', 'w')
	s_file.write(str(sober))
	s_file.close()

	d_file = open('drunk.txt', 'w')
	d_file.write(str(drunk))
	d_file.close()

runFeatureExtraction("./dax_extracted/*.txt")


