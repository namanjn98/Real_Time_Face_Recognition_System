import cv2 
import time
import os
from fnmatch import fnmatch


def getface(myname):                        #Getting face from an image 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

    in_path = os.path.realpath("getface.py")

    folder = '/real-time/Faces/%s/'%(myname)
    root = os.path.dirname(in_path) + folder 

    images_path = []
    pattern = "*.jpg"
    for path, subdirs, files in os.walk(root):  #For all images in the 'myname' directory 
    	for name in files:
    		if fnmatch(name, pattern):
    			img_root = root + name
    			img = cv2.imread(img_root)
    			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         #Color to B&W
    			faces = face_cascade.detectMultiScale(gray, 1.2, 5)  #Detecting faces, scalefactor - 1.3, neighbours - 5

    			for (x,y,w,h) in faces:
    				face_img = img[y+5:y+h-5,x+5:x+w-5]            #Cropping face image
    				face_img = cv2.resize(face_img,(32,32))          #Resizing to make uniform images
    				cv2.imwrite(img_root, face_img)                 #Saving the image back

    print "%s's Face Extracted\n"%(myname)
