import os
import pickle
import time
import math
import cvzone
import cv2
import face_recognition
import torch
from cvzone.FaceDetectionModule import FaceDetector
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import numpy as np
from datetime import datetime
##################################################
print("Checking Credentials .....")
cred = credentials.Certificate("database/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://smartattendance-c3b86-default-rtdb.firebaseio.com/",
    'storageBucket': "smartattendance-c3b86.appspot.com"
})
print("Credentials Verified")
bucket = storage.bucket()

##################################################
counter = 0
window = 0
mode = 3
imgStudent = []

confidence = 0.55
imgPath = 'database/images'
imgbg = cv2.imread('templates/bg.jpg')
folderModePath = 'templates/modes'
modePathlist = os.listdir(folderModePath)
encodePath = 'static/EncodeFile.p'
offsetPercentageW = 10
offsetPercentageH = 20
classNames = ["fake", "real"]

##################################################
imgModeList = []
# importing the mode images
for path in modePathlist:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# print(len(imgModeList))
##################################################
print("Loading Encode file .....")

file = open(encodePath, 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print(studentIds)
print("Encode file Loaded")

##################################################


def face(image):
    imgS = cv2.resize(image, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceloc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("matches ", matches)
        # print("faceDis ", faceDis)
        # exit(0)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            return studentIds[matchIndex]
        else:
            return False


def offset(xc1, xc2, yc1, yc2):
    w = xc2 - xc1
    h = yc2 - yc1
    offsetw = (offsetPercentageW / 100) * w
    xc1 = int(xc1 - offsetw)
    w = int(w + offsetw * 2)
    offseth = (offsetPercentageH / 100) * h
    yc1 = int(yc1 - offseth * 3)
    h = int(h + offseth * 3.5)
    xc2 = xc1 + w
    yc2 = yc1 + h
    return (xc1, xc2, yc1, yc2)


##################################################
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# x = torch.randn(3, 3)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x = x.to(device)
# print(f"Using device: {device}")

detector = FaceDetector()
model = YOLO("yoloModels/v5_nano-40.pt")


while True:
    new_frame_time = time.time()
    success, img = cap.read()
    # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img, bboxs = detector.findFaces(img, draw=False)
    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
            # print(x1, x2, y1, y2)

            # offset
            (x1, x2, y1, y2) = offset(x1, x2, y1, y2)
            # imgFace = img[y1:y2, x1:x2]
            # cv2.imshow("Face", imgFace)
            conf = math.ceil((box.conf[0])*100)/100
            cls = int(box.cls[0])
            #  print(classNames[cls],conf)

            if conf > confidence:
                if classNames[cls] == "real":
                    color = (0, 255, 0)
                    id = face(img)
                    out = id
                    if counter == 0:
                        counter = 1

                else:
                    out = "FAKE"
                    color = (0, 0, 255)
                    mode = 0

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cvzone.putTextRect(img,f'{out} {int(conf*100)}%', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, colorB=color, colorR=color)

    step = -1
    if counter != 0 and id != None:
        if counter == 1:
            mode = 4
            # get stufent info
            studentInfo = db.reference(f'Students/{id}').get()
            # get image from storage
            blob = bucket.get_blob(f'Orig_images/{id}.jpg')
            array = np.frombuffer(blob.download_as_string(), np.uint8)
            imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
            # Update attendance
            datetimeobject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
            secondsElapsed = (datetime.now()-datetimeobject).total_seconds()
            if secondsElapsed > 30:
                print(studentInfo)
                ref = db.reference(f'Students/{id}')
                studentInfo['total_attendance'] += 1
                ref.child('total_attendance').set(studentInfo['total_attendance'])
                ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            else:
                print("Already Marked ")
                mode = 2
                step = 1
                counter = -1

        counter += 1
        if counter == 10:
            mode = 1

        if counter == 15:
            mode = 3
            counter = 0
            step = 0
            studentInfo = []
            imgStudent = []

    elif step == 1:
        mode = 2
    else:
        mode = 3
    imgbg[100:100 + 480, 120:120 + 640] = img
    imgbg[0:720, 900:900 + 350] = imgModeList[mode]

    if mode == 4:
        cv2.putText(imgbg, str(studentInfo['total_attendance']), (950, 106), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 1)
        (w, h), _ = cv2.getTextSize(f"Name : {studentInfo['name']}", cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        off = (311-w)//2
        cv2.putText(imgbg, f"Name :{studentInfo['name']}", (910+off, 460), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 0, 0), 1)
        cv2.putText(imgbg, f"ID : {id}", (925, 530), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                    (255, 255, 255), 1)
        cv2.putText(imgbg, f"Standing : {studentInfo['standing']}", (930, 630), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 1)

        imgbg[182:182+225, 922:922+290] = imgStudent

    #  cv2.imwrite("non-static/live/live.jpg", img)
    cv2.imshow("Face Attendance", imgbg)
    cv2.waitKey(1)
