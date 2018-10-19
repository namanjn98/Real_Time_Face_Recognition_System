import numpy as np
import cv2
import model
import getface
import capture
import test
import os
import pandas as pd
import sys

name = sys.argv[1]

# try:
capture.capture(name)					#Capture Photos
getface.getface(name)					#Get Face from the Photos

images, len_images, size, images_path = model.images('real-time')		
mean_vec = model.mean_vec(images, size, len_images)
nor_images = model.normalise(images,mean_vec)

k = (len(nor_images)/2) + 1
eig_vec, weights = model.pca(nor_images, k)					#Using PCA to get Eigenfaces 

np.save('real-time/vectors/mean_vec',mean_vec)              #Storing mean_vec for real time
np.save('real-time/vectors/eig_vec',eig_vec)				#Storing eig_vec for real time
np.save('real-time/vectors/weights',weights)				#Storing weights for real time

pd.DataFrame(images_path).to_csv('real-time/vectors/image_path.csv', index = False)  #Image paths for matching the index of nearest face class

# except:
# 	print 'Click the photos again\n'

