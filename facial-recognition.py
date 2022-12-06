from deepface import DeepFace
from retinaface import RetinaFace
from deepface.basemodels import VGGFace
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import cv2

def getFaces(img_path):

    """
    input: exact path to image
    output: list of size N-by-5, containing four image coords representing bounding box of face,
    and a string representing the closest identity in the database for each detected face. 
    """
    # create dictionary to store facial regions
    face_dict = RetinaFace.detect_faces(img_path)

    # convert dict to pandas dataframe
    face_df = pd.DataFrame.from_dict(face_dict).T

    # store group image as numpy array
    img = cv2.imread(img_path)

    output = []
    faces = []

    # loop over detected faces
    for ind in face_df.index:
        # get bounding boxes for each face
        rect = face_df['facial_area'][ind]

        # extract faces by getting sub image for each face from group imgage
        ROI = img[rect[1]:rect[3], rect[0]:rect[2]]
        faces.append(ROI)
        output.append(rect)


    BGRfaces = []

    for face in faces:
        # convert sub image to numpy array, convert to BGR
        BGRfaces.append(cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR))

    for i in range(len(BGRfaces)):
        df = DeepFace.find(img_path = BGRfaces[i], db_path= "C:\python-projects\ece-523-final-proj\database", enforce_detection = False, silent = "True", prog_bar = "False")
        id_path = df.iloc[0].identity
        norm_path = os.path.normpath(id_path)
        split_path = norm_path.split(os.sep)

        # index of split path is name of id dir in database, e.g. root\my-project\database\<id>\myimg.png
        id = split_path[4]
        output[i].append(id)
    
    return output