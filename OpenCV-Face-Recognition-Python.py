# test
# Face Recognition with OpenCV
# hey
# To detect faces, I will use the code from my previous article on [face detection](https://www.superdatascience.com/opencv-face-detection/). So if you have not read it, I encourage you to do so to understand how face detection works and its Python coding. 

import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Labels for our data
subjects = [0, 1, 2, 3, 4, 5, 6, 7]

#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('C:\python-projects\opencv-face-recognition-python\opencv-files/haarcascade_frontalface_default.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


def detect_eyes(gray):
    eye_cascade = cv2.CascadeClassifier('C:\python-projects\opencv-face-recognition-python\opencv-files\haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 5)

    if (len(eyes) < 2):
        return None, None
    
    return eyes[0], eyes[1]


def align_face(face):
    left_eye, right_eye = detect_eyes(face)

    if all(item is None for item in (left_eye, right_eye)):
        return None

    (xl, yl, wl, hl) = left_eye
    (xr, yr, wr, hr) = right_eye

    left_eye_y = yl + hl//2
    left_eye_x = xl + wl//2
    right_eye_y = yl + hl//2
    right_eye_x = xl + wl//2

    eyes_dx = abs(left_eye_x - right_eye_x)
    eyes_dy = abs(left_eye_y - right_eye_y)

    # Prevent divide-by-zero errors
    if (eyes_dx == 0):
        eyes_dx = eyes_dx + 0.001

    # Calculate rotation angle
    angle = np.arctan(eyes_dy / eyes_dx)

    if left_eye_y >  right_eye_y:
        # Rotate counter-clockwise
        aligned_face = Image.fromarray(face).rotate(angle)

    if left_eye_y <=  right_eye_y:
        # Rotate clockwise
        aligned_face = Image.fromarray(face).rotate(-angle)

    # The following two lines are used to save the aligned and resized face to file
    # path = 'C:/python-projects/opencv-face-recognition-python/aligned-images' + image_name + '.jpg'
    # aligned_face.resize((200,200)).save(path, 'JPEG')
    aligned_face = np.array(aligned_face.resize((200,200)))
    return aligned_face


# Function that reads all training images, detects face from each image, and will return a list of faces and a list of labels for each face.
def prepare_training_data(data_folder_path):

    # Get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    # List to hold all faces
    faces = []
    # List to hold labels for all subjects
    labels = []
    
    # Go through each directory; read images
    for dir_name in tqdm(dirs):
            
        # Extract label number of subject from dir_name (format of dir name = s[label], so removing letter 's' from dir_name will give us the label)
        label = int(dir_name.replace("s", ""))
        
        # Build path of directory 
        subject_dir_path = data_folder_path + "/" + dir_name
        
        # Get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        # Go through each image name, read image, detect face and add face to list of faces
        for image_name in tqdm(subject_images_names):
            
            # Ignore system files like .DS_Store
            if image_name.startswith("."):
                continue
            
            # Build image path
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)
            
            cv2.waitKey(100)
            
            # Detect face
            face, rect = detect_face(image)

            # Align face
            aligned_face = align_face(face)

            # Ignore faces that are not detected
            if aligned_face is not None:
                #add face to list of faces
                faces.append(aligned_face)
                #add label for this face
                labels.append(label)
    
    return faces, labels

# Prepare the training data.  One list will contain all the faces and other list will contain respective labels for each face
faces, labels = prepare_training_data('C:/python-projects/opencv-face-recognition-python/training-data')

# Print total amount faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# As an alternative to the eigenfaces recognizer, fisherfaces can be used by replaceing the line below with:
#face_recognizer = cv2.face.FisherFaceRecognizer_create()

# EigenFaceRecognizer 
face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Train our face recognizer
face_recognizer.train(faces, np.array(labels))



# This function recognizes the person in image passed
def predict(test_img):
    # Make a copy of the image as we don't want to chang original image
    img = test_img.copy()

    # Detect face from the image
    face, rect = detect_face(img)

    aligned_face = align_face(face)

    # Predict the image using our face recognizer
    if aligned_face is not None:
        label, confidence = face_recognizer.predict(aligned_face)
        #get name of respective label returned by face recognizer
        print("Label: " + str(label))
    return img

# Perform predictions
# Load test images
test_data_path = 'C:/python-projects/opencv-face-recognition-python/test-data'
test_dir = os.listdir('C:/python-projects/opencv-face-recognition-python/test-data')
for img_name in test_dir:
    test_img_path = test_data_path + '/' + img_name
    test_img = cv2.imread(test_img_path)
    predict(test_img)






