import cv2
import os
import face_recognition
import pickle
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db

cred = credentials.Certificate("D:\Projects\Face_Recognization_with_real_time_database\serviceForDatabase.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://faceattendacerealtime-7a11f-default-rtdb.firebaseio.com/",
    'storageBucket':"faceattendacerealtime-7a11f.appspot.com"
},name='newDatabaseApp')


app = firebase_admin.get_app(name="newDatabaseApp")

ref = db.reference('/',app,"https://faceattendacerealtime-7a11f-default-rtdb.firebaseio.com/")

data = {
    "12345":
    {
        "name" : "Mukul Kandpal",
        "Major" : "Machine Learning",
        "Starting Year" : "2020",
        "Total Attendance" : 14,
        "Standing" : 14,
        "Year" : 3,
        "Last Attendace Time" : "2023-02-21 00:54:34"
    },
    "82183":
    {
        "name" : "M.S.Dhoni",
        "Major" : "Cricket",
        "Starting Year" : "2019",
        "Total Attendance" : 17,
        "Standing" : 13,
        "Year" : 2,
        "Last Attendace Time" : "2023-02-21 00:54:34"
    },
    "83940":
    {
        "name" : "Elon Musk",
        "Major" : "Rocket Science",
        "Starting Year" : "2022",
        "Total Attendance" : 12,
        "Standing" : 9,
        "Year" : 2,
        "Last Attendace Time" : "2023-02-21 00:54:34"
    },
    "97592":
    {
        "name" : "Sachin Tendulkar",
        "Major" : "Batsman",
        "Starting Year" : "2023",
        "Total Attendance" : 12,
        "Standing" : 5,
        "Year" : 1,
        "Last Attendace Time" : "2023-02-21 00:54:34"
    }
}

for key,value in data.items():
    ref.child(key).set(value)



from firebase_admin import storage

# importing the images
folderPath = 'D:\Projects\Face_Recognization_with_real_time_database\Images'

PathList = os.listdir(folderPath)

imgList = []

# storing id's of every person
studentId = []

for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))

    # using splitext function to split the name and .png part of the image as name is the id
    studentId.append(os.path.splitext(path)[0])



# importing the modes
folderModePath = 'D:\Projects\Face_Recognization_with_real_time_database\Resources\Modes'

modePathList = os.listdir(folderModePath)

imgModeList = []

for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# print(len(imgModeList))


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


print("Encoding started...")
encodeListKnown = findEncodings(imgList)
print("Encoding ended")



encodeListKnownWithIds = [encodeListKnown, studentId]
file = open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()


file = open('EncodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentId = encodeListKnownWithIds
print("loading completed")
print(studentId)

# activating webcam
cap = cv2.VideoCapture()

# importing background image
imgBackground = cv2.imread(r'D:\Projects\Face_Recognization_with_real_time_database\Resources\background1.png')

# setting size of height and width of the image and webcam
cap.set(3, 640)
cap.set(4, 480)


modeType = 0





while True:
    # capturing live image . Here webcam is started it is not just visible to us but functioning is started
    res, img = cap.read()

    # resizing image because small image works faster
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    # marking out face from the whole image
    faceCurrFrame = face_recognition.face_locations(imgs)

    # encoding the face of the live webcame image
    encodeCurrFrame = face_recognition.face_encodings(imgs, faceCurrFrame)

    # overlapping the background image and webcam
    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    # matching the encodings of the live face captured with the database images
    for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        # finding the distance of live image with each image in database
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("matches: ", matches)
        # print("distance: ", faceDis)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
        # print("known face detected")
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

    cv2.imshow("Face Detection", imgBackground)
    if cv2.waitKey(10) == ord("a"):
        break
cap.release()