import cv2
import sys
import urllib.request as rq
import numpy as np

def urltoimg(url):
    resp = rq.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def face_detect(imagePath):
    # Get user supplied values
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    glassesCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    # Read the image
    image = urltoimg(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    eyes = eyeCascade.detectMultiScale(rgb,
                                       scaleFactor=1.1,
                                       minNeighbors=5,
                                       minSize=(60, 60),
                                       flags=cv2.CASCADE_SCALE_IMAGE)
    glasses = glassesCascade.detectMultiScale(rgb,
                                              scaleFactor=1.1,
                                              minNeighbors=5,
                                              minSize=(60, 60),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
    EyesOpen=False
    GlassesWearing=False
    if(len(eyes)==0):
        EyesOpen=False
    else:
        EyesOpen=True

    if(len(glasses)==0):
        GlassesWearing=False
    else:
        GlassesWearing=True

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    #for (x, y, w, h) in faces:
       #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #cv2.imshow("Faces found", image)
    #cv2.waitKey(0)
    return image,len(faces),EyesOpen,GlassesWearing