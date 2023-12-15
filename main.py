import base64

from flask import Flask, request, jsonify
from flask_cors import CORS
import extracting_features
import face_detect
import facial_recognition
import pymongo
import sys


def ConnectDB():
    try:
        connection ="mongodb+srv://usama:112113114@cluster0.u1c4y.mongodb.net/IDVerify?retryWrites=true&w=majority"
        client = pymongo.MongoClient(connection)
        dblist = client.list_database_names()
        if "IDVerify" in dblist:
            print("The database exists.")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise


app = Flask(__name__)
CORS(app)


auth_keys = {"NIN@FACE", "Fahad", "Usama", "Ishaq","2jP8BqqMdgSqU50oI9CP94Y75SS1RyrX5i"}


def authenticate():
    headers = request.headers
    auth = headers.get("Api-Key")
    if auth in auth_keys:
        return True
    else:
        return False


@app.route('/test', methods=['GET'])
def hello_world():
    if authenticate():
        return {"value": "running is ok"}
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401


"""
@api {POST} /face_detect  Detect If Face is present in picture
@apiName Detect Face
@apiGroup Facial Recognition

@apiParam {image_path} Webpath of the Image.

@apiSuccess {String} answer True of False(weather face is present or not).
@apiSuccess {int} Number of Faces  Number of faces in picture.
"""


@app.route('/face_detect', methods=["POST"])
def face_detects():
    if authenticate():
        accept = False
        data = request.json
        image_path = data['path']
        image, number_of_faces,eyes_Open,Wearing_Glasses = face_detect.face_detect(image_path)
        if number_of_faces == 1: accept = True
        returndata = {"answer": accept, "Number of Faces": number_of_faces,"Eyes are open":eyes_Open,"Wearing_Glasses":Wearing_Glasses}
        return returndata
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401


"""
@api {POST} /extract_features  Extract features of face in picture and save it in a file locally
@apiName Extract Features
@apiGroup Facial Recognition

@apiParam {front_pic} Webpath of the Front face Image.
@apiParam {left_pic} Webpath of the Left face Image.
@apiParam {right_pic} Webpath of the right face Image.
@apiParam {Did} Database ID of the user.

@apiSuccess {filename} name of the file where face features are stored.
"""
def isBase64(s):
    try:
        f=base64.b64decode(s)
        g=base64.b64encode(f)
        return str(base64.b64encode(base64.b64decode(s))) == s
    except Exception:
        return False

@app.route('/extract_features', methods=["POST"])
def extract_features():
    if authenticate():
        data = request.json
        front_pic = data['front_pic']
        left_pic = data['left_pic']
        right_pic = data['right_pic']
        Did = data['Did']
        dumpdata, dump_filename = extracting_features.extract_facial_features(front_pic, left_pic, right_pic, Did)
        returndata = {"filename": dump_filename}
        return returndata
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401

@app.route('/extract_features_Single', methods=["POST"])
def extract_features_single():
    if authenticate():
        data = request.json
        pic = data['Path_for_Picture']

        Did = data['Did']
        dumpdata, dump_filename = extracting_features.extract_facial_features_Single(pic, Did)
        returndata = {"filename": dump_filename}
        return returndata
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401
"""
@api {POST} /face_recognize  Match the picture recived from the NIN database to the ones provided by the user
@apiName Face recognition
@apiGroup Facial Recognition

@apiParam {image_path} base64 image string.
@apiParam {Did} Database ID of the user.

@apiSuccess {filename} name of the file where face features are stored.
"""


@app.route('/face_recognize', methods=["POST"])
def face_recognize():
    if authenticate():
        data = request.json
        image_path = data['image_path']
        NIN_recieved = data['Did']
        NIN_calculated, confidence, livelinessCheck = facial_recognition.face_recognize(image_path)
        if not NIN_calculated:

            return {"result": False}

        else:
            if (NIN_recieved == NIN_calculated[0]):
                return {"ID": NIN_calculated[0], "result": True,"confidence":confidence,"Liveliness Check":livelinessCheck}
            else:
                return {"result": False,"confidence":confidence,"Liveliness Check":livelinessCheck}
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401


@app.route('/face_Recognize_Using_BASE64', methods=["POST"])
def face_detects_using_base64():
    if authenticate():
        data = request.json
        image1 = data['Image1']
        image2 = data['Image2']
        DID = data['DID']
        try:
            filename,Pic1URL = extracting_features.extract_facial_features_Single_BASE64(image1,DID)
            match, confidence, livelinessCheck,Pic2URL = facial_recognition.face_recognizeBase64(image2,filename)
            return {"Match": match, "confidence": confidence,"liveliness_check":livelinessCheck,"Image1URL":Pic1URL,"Image2URL":Pic2URL}
        except:
            return jsonify({"message": "ERROR: Unrecognized String"}), 408
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401


