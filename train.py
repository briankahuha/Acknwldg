import os
import cv2 as cv
import numpy as np
import pickle
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "SampleImages")
schID = str(651436)

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
