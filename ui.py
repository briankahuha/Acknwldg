import sys
import os
import pickle
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
import cv2 as cv
import numpy as np
from PIL import Image
import csv
import time
import pyrebase
import cProfile
import pstats
from pstats import SortKey


firebaseConfig = {
    "apiKey": "AIzaSyA_VFqAxvUDdfhzYU_SfC-DpnQNjxabUyc",
    "authDomain": "facedet-c5639.firebaseapp.com",
    "databaseURL": "https://facedet-c5639.firebaseio.com",
    "projectId": "facedet-c5639",
    "storageBucket": "facedet-c5639.appspot.com",
    "messagingSenderId": "902109850304",
    "appId": "1:902109850304:web:457b4a53aecc7c5c945b25"
  }

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

auth = firebase.auth()



class Login(QDialog):
    def __init__(self):            
        super(Login, self).__init__()  
        loadUi("login.ui", self)
        self.loginbtn.clicked.connect(self.gotomainpage)
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.createaccbtn.clicked.connect(self.gotocreate)
        self.invalid.setVisible(False)

    def loginfunction(self):
        email = self.email.text()
        password = self.password.text()
        try:
            auth.sign_in_with_email_and_password(email, password)
        except:
            self.invalid.setVisible(True)
        
    def gotocreate(self):
        createacc = CreateAcc()
        widget.addWidget(createacc)
        widget.setCurrentIndex(widget.currentIndex()+1)
    
    def gotomainpage(self):
        mainpage = MainPage()
        widget.addWidget(mainpage)
        widget.setCurrentIndex(widget.currentIndex()+1)

class CreateAcc(QDialog):
    def __init__(self):
        super(CreateAcc, self).__init__()
        loadUi("createacc.ui", self)
        self.signupbtn.clicked.connect(self.createaccfunction)
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.confirmpass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.invalid.setVisible(False)

    def createaccfunction(self):
        email = self.email.text()
        if self.password.text() == self.confirmpass.text():
            password = self.password.text()
            try:
                auth.create_user_with_email_and_password(email, password)
                login = Login()
                widget.addWidget(login)
                widget.setCurrentIndex(widget.currentIndex()+1)
            except:
                self.invalid.setVisible(True)

class MainPage(QDialog):
    def __init__(self):
        super(MainPage, self).__init__()
        loadUi("mainpage.ui", self)
        self.picturebtn.clicked.connect(self.capture)
        self.attendance.clicked.connect(self.trainnverify)
        self.record.clicked.connect(self.takeattendance)
        self.trained.setVisible(False)
        
    def capture(self):
        schID = self.schid.text()
        name = self.name.text()
        major = self.major.text()
        currentyr = self.currentyr.text()
        pnumber = self.number.text()

        db.child("students").push(data={"School ID":schID, "Name":name, "Major":major, "Current Year":currentyr, "Phone Number":pnumber})

        cam = cv.VideoCapture(0, cv.CAP_DSHOW)
        cv.namedWindow("Capture")

        img_counter = 0

        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            cv.imshow("test", frame)

            k = cv.waitKey(1)
            if k%256 == 27:
                break
            if k%256 == 32:
                while img_counter<=40: 
                    img_counter += 1
                    img_name = "{}.png".format(img_counter)
                    cv.imwrite("SampleImages\ "+schID+"-"+img_name, frame)
                    

        cam.release()
        self.oneTest()

    def trainnverify(self):
        self.train()
        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        recognizer = cv.face.LBPHFaceRecognizer_create()
        recognizer.read("trainner.yml")

        labels = {}
        with open("labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v:k for k,v in og_labels.items()}

        cap = cv.VideoCapture(0, cv.CAP_DSHOW)

        while True:
            ret, frame = cap.read()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                #roi_color = frame[y:y+h, x:x+w]

                id_, conf = recognizer.predict(roi_gray)
                if conf>=45 and conf<=85:
                    print(id_)
                    print(labels[id_])
                    font = cv.FONT_HERSHEY_SIMPLEX
                    name = labels[id_]
                    color = (255, 255, 255)
                    stroke = 2
                    cv.putText(frame, name, (x,y), font, 1, color, stroke, cv.LINE_AA)
                elif conf<45:
                    font = cv.FONT_HERSHEY_SIMPLEX
                    name = "Unidentified Person"
                    color = (255, 255, 255)
                    stroke = 2
                    cv.putText(frame, name, (x,y), font, 1, color, stroke, cv.LINE_AA)
                #img_item = "my-image.png"
                #cv.imwrite(img_item, roi_gray)
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            cv.imshow('frame', frame)

            k = cv.waitKey(1)
            if k%256 == 27:
                break

        cap.release()
        cv.destroyAllWindows()

        self.oneTest()

    def train(self):
        schID = self.schid.text()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR, "SampleImages")

        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        recognizer = cv.face.LBPHFaceRecognizer_create()

        current_id = 0
        label_ids = {}
        y_labels = []
        x_train = []


        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(root, file)
                    label = os.path.basename("SampleImages\\"+schID+"-1.png").split("-1.png")[0].split("\\")[-1:][0]
                    if label in label_ids:
                        pass
                    else:
                        label_ids[label] = current_id
                        current_id += 1
                        id_ = label_ids[label]
                    #y_labels.append(label)
                    #x_train.append(path)
                    pil_image = Image.open(path).convert("L")
                    image_array = np.array(pil_image, "uint8")
                    faces = face_cascade.detectMultiScale(image_array, 1.1, 4)

                    for (x, y, w, h) in faces:
                        roi = image_array[y:y+h, x:x+w]
                        x_train.append(roi)
                        y_labels.append(id_)

        with open("labels.pickle", 'wb') as f:
            pickle.dump(label_ids, f)

        recognizer.train(x_train, np.array(y_labels))
        recognizer.save("trainner.yml")

        print("images trained")
        self.trained.setVisible(True)

        self.oneTest()

    def takeattendance(self):
        now = time.time()
        datentime = time.ctime(now)
        schID = self.schid.text()
        name = self.name.text()
        headers = ["Name", "School ID", "Date and Time"]
        row = [name, schID, datentime]
        with open("attendance.csv", "a+") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerow(row)
        csvfile.close()

        db.child("attendance").push(data={ "Name":name, "School ID":schID, "Date and Time":datentime})

        self.oneTest()

    def oneTest(self):
        cProfile.run("app.exec_()", "output.dat")

        with open("output_time.txt", "a+") as f:
            p = pstats.Stats("output.dat", stream=f)
            p.sort_stats("time").print_stats()

        with open("output_calls.txt", "a+") as f:
            p = pstats.Stats("output.dat", stream=f)
            p.sort_stats("calls").print_stats()
 

app =  QApplication(sys.argv)
mainwindow = Login()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedHeight(620)
widget.setFixedWidth(480)
widget.show()
app.exec_()
