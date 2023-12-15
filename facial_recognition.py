import face_recognition
import imutils
import pickle
import requests
import base64
import urllib.request as rq
import numpy
from PIL import Image
import cv2
from io import BytesIO
import numpy as np
import os


def readb64(base64_string):
    sbuf = BytesIO()
    base64_string = base64_string.encode("ascii", errors="ignore").decode()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return pimg


def imgto64(img):
    return (img.encode("base64"))


def urltoimg(url):
    resp = rq.urlopen(url)

    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def face_recognize(path_for_picture):
    # find path of xml file containing haarcascade file
    cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    glassesCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # load the known faces and embeddings saved in last file
    data = pickle.loads(open("face_enc", "rb").read())
    # Find path to the image you want to detect face and pass it here
    # image=urltoimg(path_for_picture)

    image = readb64(path_for_picture)
    open_cv_image = numpy.array(image)

    # open_cv_image = cv2.imread(path_for_picture, 0)

    rgb = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    # convert image to Greyscale for haarcascade
    # gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(rgb,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

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

    livelinessCheck = False
    if (len(eyes) == 0 and len(glasses) == 0):
        livelinessCheck = False
    else:
        livelinessCheck = True
    # the facial embeddings for face in input
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    encodings2 = face_recognition.face_encodings(rgb)
    names = []
    confidence = 0
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
        # Compare encodings with encodings in data["encodings"]
        # Matches contain array with boolean values and True for the embeddings it matches closely
        # and False for rest
        confidence_level = face_recognition.face_distance(data["encodings"], encoding)
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding, 0.5)

        confidence = (1 - min(confidence_level)) * 100
        # set name =inknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            # Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                # Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                # increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
                # set name which has highest count
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)
            # loop over the recognized faces

        # cv2.imshow("Frame", open_cv_image)
        # cv2.waitKey(0)

    print(names)
    return names, confidence, livelinessCheck


def face_recognizeBase64(path_for_picture, filename):
    # find path of xml file containing haarcascade file
    cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    glassesCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # load the known faces and embeddings saved in last file
    data = pickle.loads(open(filename, "rb").read())
    # Find path to the image you want to detect face and pass it here
    # image=urltoimg(path_for_picture)
    try:
        image = readb64(path_for_picture)
        open_cv_image = numpy.array(image)

        URL = "https://nimc-service.herokuapp.com/api/upload-image-service"
        PARAMS = {'selfieImage': path_for_picture}
        r = requests.post(url=URL, data=PARAMS)
        datan = r.json()
        PicURL = datan['message']
        # open_cv_image = cv2.imread(path_for_picture, 0)

        rgb = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

        # convert image to Greyscale for haarcascade
        # gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(rgb,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

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

        livelinessCheck = False
        if (len(eyes) == 0 and len(glasses) == 0):
            livelinessCheck = False
        else:
            livelinessCheck = True
        # the facial embeddings for face in input
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)
        encodings2 = face_recognition.face_encodings(rgb)
        names = []
        confidence = 0
        # loop over the facial embeddings incase
        # we have multiple embeddings for multiple fcaes
        for encoding in encodings:
            # Compare encodings with encodings in data["encodings"]
            # Matches contain array with boolean values and True for the embeddings it matches closely
            # and False for rest
            confidence_level = face_recognition.face_distance(data["encodings"], encoding)
            matches = face_recognition.compare_faces(data["encodings"], encoding, 0.5)
            match = False
            confidence = (1 - min(confidence_level)) * 100
            # set name =inknown if no encoding matches
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                match = True
                # Find positions at which we get True and store them
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    # Check the names at respective indexes we stored in matchedIdxs
                    name = data["names"][i]
                    # increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                    # set name which has highest count
                    name = max(counts, key=counts.get)

                # update the list of names
                names.append(name)
                # loop over the recognized faces

            # cv2.imshow("Frame", open_cv_image)
            # cv2.waitKey(0)

        print(names)
        return match, confidence, livelinessCheck, PicURL
    except:
        return None
