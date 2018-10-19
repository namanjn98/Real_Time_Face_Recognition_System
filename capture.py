import cv2
import os

def capture(name):                  #To capture photos - (name) is of the face 
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("New_Face")

    img_counter = 0

    while True:
        ret, frame = cam.read()

        cv2.imshow("New_Face", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed for closing 

            print "Closing Camera...\n"
            break

        elif k%256 == 32:
            # SPACE pressed for clicking photo

            img_name = "{}_frame_{}.jpg".format(name, img_counter)  

            in_path = os.path.realpath("capture.py")
            folder = '/real-time/Faces/%s/'%(name)
            root = os.path.dirname(in_path) + folder

            try:
                os.makedirs(root)                               #Making folder for a new face
                cv2.imwrite(root + img_name, frame)
            except:
                cv2.imwrite(root + img_name, frame)

            img_counter += 1
            print "Photo Clicked - %d\n"%(img_counter) 

    cam.release()

    cv2.destroyAllWindows()