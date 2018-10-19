import cv2
import numpy
from matplotlib import pyplot as plt
from numpy import linalg as LA
from scipy.spatial import distance
import os
from fnmatch import fnmatch


def test_img(mean_vec, eig_vec, weights, input_img):  #Processing unknown face for face class
	input_img = input_img.flatten()
	mean_input = input_img - mean_vec

	w_main = []										  #Weights of the unknown image projected on eigenfaces
	for u in eig_vec:
	    w = numpy.matmul(u.transpose(),mean_input)
	    w_main.append(w)
	
	dist = []										#List of distances from weights of training set and unknown face
	for in_w in weights:
		ed = distance.euclidean(w_main, in_w)
		dist.append(ed)

	min_dist_index = numpy.argmin(dist) #Argument index with minimum distance
	min_dist = min(dist)				#Minimum Distance

	if min_dist > 1000:				#Threshold for a face class
		min_dist = -1

	return min_dist_index, min_dist
