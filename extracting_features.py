import string
import string
import requests
from imutils import paths
import face_recognition
import pickle
import cv2
import urllib.request as rq
import os
from io import BytesIO
import base64
from PIL import Image
import numpy as np


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


def extract_facial_features(frontPic, LeftPic, RightPic, Did):
    # get paths of each file in folder named Images
    # Images here contains my data(folders of various persons)
    knownEncodings = []
    knownNames = []
    # loop over the image paths
    image = urltoimg(frontPic)
    # image = cv2.imread(frontPic, 0)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(Did)

    image = urltoimg(LeftPic)
    # image = cv2.imread(LeftPic, 0)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(Did)

    image = urltoimg(RightPic)
    # image = cv2.imread(RightPic, 0)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(Did)

    data = {"encodings": knownEncodings, "names": knownNames}
    # use pickle to save data into a file for later use
    filename = "face_enc"
    f = open(filename, "wb")
    dumpdata = pickle.dumps(data)
    f.write(pickle.dumps(data))
    f.close()
    return dumpdata, filename


def extract_facial_features_Single(Pic, Did):
    # get paths of each file in folder named Images
    # Images here contains my data(folders of various persons)
    knownEncodings = []
    knownNames = []
    # loop over the image paths
    image = urltoimg(Pic)
    # image = cv2.imread(Pic, 0)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(Did)

    data = {"encodings": knownEncodings, "names": knownNames}
    # use pickle to save data into a file for later use
    filename = "face_enc"
    f = open(filename, "wb")
    dumpdata = pickle.dumps(data)
    f.write(pickle.dumps(data))
    f.close()
    return dumpdata, filename


def extract_facial_features_Single_BASE64(Pic, Did):
    # get paths of each file in folder named Images
    # Images here contains my data(folders of various persons)
    knownEncodings = []
    knownNames = []
    # loop over the image paths
    # image = cv2.imread(Pic, 0)
    URL = "https://nimc-service.herokuapp.com/api/upload-image-service"
    PARAMS = {'selfieImage': Pic}
    r = requests.post(url=URL, data=PARAMS)
    data = r.json()
    PicURL = data['message']
    try:
        image = readb64(Pic)
        image = np.array(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(Did)

        data = {"encodings": knownEncodings, "names": knownNames}
        # use pickle to save data into a file for later use
        filename = "face_enc" + Did
        f = open(filename, "wb")
        dumpdata = pickle.dumps(data)
        f.write(pickle.dumps(data))
        f.close()
        return filename, PicURL
    except:
        return None
