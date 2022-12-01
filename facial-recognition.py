from deepface import DeepFace
from retinaface import RetinaFace
from deepface.basemodels import VGGFace
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import cv2

# model =VGGFace.loadModel

# return dict for each face in image
face_dict = RetinaFace.detect_faces(img_path = "img1.png")

face_df = pd.DataFrame.from_dict(face_dict).T

# print(face_df)
# print(face_df['facial_area'][0][0])

img = cv2.imread("img1.png")
# rect = face_df['facial_area'][0]


output = {}

faces = []

for ind in face_df.index:
    rect = face_df['facial_area'][ind]
    ROI = img[rect[1]:rect[3], rect[0]:rect[2]]
    faces.append(ROI)
    
    

# faces = RetinaFace.extract_faces(img_path = "img1.png", align = True)

BGRfaces = []

for face in faces:
    BGRfaces.append(cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR))

# plt.imshow(BGRfaces[6])
# plt.show()


for i in range(len(BGRfaces)):
    df = DeepFace.find(img_path = BGRfaces[i], db_path= "database", enforce_detection = False, silent = "True", prog_bar = "False")
    id_path = df.iloc[0].identity
    norm_path = os.path.normpath(id_path)
    split_path = norm_path.split(os.sep)

    # index of split path is name of id dir in database, e.g. root\my-project\database\<id>\myimg.png
    id = split_path[4]

print(output)   





    #  facial_area = identity["facial_area"]

    # for dir in split_path:
    #     if dir == "Ahmed-A":
    #         id = 0
    #     elif dir == "s1":
    #         id = 1
    #     elif dir == "s2":
    #         id = 2
    #     elif dir == "s3":
    #         id = 3
    #     elif dir == "s4":
    #         id = 4
    #     elif dir == "s5":
    #         id = 5
    #     elif dir == "s6":
    #         id = 6
    #     elif dir == "s7":
    #         id = 7

    #     print(id)

# obj = RetinaFace.detect_faces("img1.png")

# print(len(obj.keys()))

# for key in obj.keys():
#     identity = obj[key]
#     print(identity)

#     facial_area = identity["facial_area"]