import cv2
import numpy
import test
import pandas as pd
import os

mean_vec = numpy.load('real-time/vectors/mean_vec.npy') #Loading training mean_vec
eig_vec = numpy.load('real-time/vectors/eig_vec.npy')	#Loading training eig_vec
weights = numpy.load('real-time/vectors/weights.npy')	#Loading training weights

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read() 									#Input Video Stream
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)			#Color ot B&W
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)		#Detecting Face

    for (x,y,w,h) in faces:
    	face_img = img[y:y+h,x:x+w]						
    	face_img = face_img[:,:,0]
    	face_img = cv2.resize(face_img,(32,32))

    	ind, dis = test.test_img(mean_vec, eig_vec, weights, face_img)				 #Checking the image class 
    	if dis != -1:
    		list_images = list(pd.read_csv('real-time/vectors/image_path.csv')['0']) #For Indexing 
    		text_name = os.path.basename(os.path.dirname(list_images[ind])) 		 #For Name of the face class
    	else:
    		text_name = ''

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)								 #Putting rectangle on face
        cv2.putText(img, text_name, (x, y-2), cv2.FONT_HERSHEY_PLAIN, 2,(0, 0, 255), 2) #Putting name on face



    cv2.imshow('Main',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()