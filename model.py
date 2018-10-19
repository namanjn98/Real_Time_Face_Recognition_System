import cv2
import numpy
from matplotlib import pyplot as plt
from numpy import linalg as LA
from scipy.spatial import distance
import os
from fnmatch import fnmatch

def images(dataset):
	if dataset == 'preset':			#For accuracy on a dataset
		folder = '/preset/'

	elif dataset == 'real-time': 	#For real-time
		folder = '/real-time/'		

	in_path = os.path.realpath("train.py")             
	root = os.path.dirname(in_path) + folder           

	pattern = "*.jpg"

	images_path = []

	for path, subdirs, files in os.walk(root):		#collecting all images for training
	    for name in files:
	        if fnmatch(name, pattern):
	            images_path.append(os.path.join(path, name))

	images = []
	for img_path in images_path:
	    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  #Grayscale input
	    for_img_size = img

	    img = img.flatten() #Flatten to M*N*1 vector
	    images.append(img) 

	size = for_img_size.size #size of image

	return images, len(images), size, images_path 



def mean_vec(images, size, m):			#Getting mean_vec of training set
	sum_input = [0 for i in range(size)]
	for img in images:
		sum_input += img 

	mean_input = sum_input/float(m)

	return mean_input



def normalise(images, mean_vec): #Normalising images with mean_vec
	nor_images = []
	for img in images:
	    img = img - mean_vec
	    nor_images.append(img)

	return nor_images



def pca(nor_images, k):					#PCA Function
	A = numpy.transpose(numpy.matrix(nor_images)) 		#A = [img1 img2 img3 ...]
	mat_pseudo = numpy.matmul(A.transpose(),A) #### M*M
	eig_val, v = LA.eig(mat_pseudo)			#v = eigen vectors for A(trans)*A

	M = len(nor_images)

	eig_vec = []
	c = 0

	for m in range(M):
	    if c < k:
	        u = numpy.matmul(A,v[:,m].real)		#Getting best k eigen vectors
	        x = numpy.linalg.norm(u)			#Normalising ||u|| = 1
	        u_ = u/x
	        
	        eig_vec.append(u_)
	        c += 1
	    else:
	        break

	weights = []							#Getting weights of all the training set
	for img in nor_images:
		w_main = []
		for u in eig_vec:
		    w = numpy.matmul(u.transpose(), img)
		    w_main.append(w)

		weights.append(numpy.array(w_main))

	return eig_vec, weights