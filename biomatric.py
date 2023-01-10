import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import csv



#loading model

# Step 1: load model
yunet = cv.FaceDetectorYN.create(
  model="face_detection_yunet_2022mar.onnx",
  config="",
  input_size=[480, 640], # [width, height]
  score_threshold=0.99,
  backend_id=cv.dnn.DNN_BACKEND_DEFAULT, # optional
  target_id=cv.dnn.DNN_TARGET_CPU, # optional
)
sface = cv.FaceRecognizerSF.create(
  model="face_recognition_sface_2021dec.onnx",
  config="",
  backend_id=cv.dnn.DNN_BACKEND_DEFAULT, # optional
  target_id=cv.dnn.DNN_TARGET_CPU, # optional
)

# Step 2: load image
img1 = cv.imread("lena.jpg")
img2 = cv.imread("lena2.jpg")

# Step 3: detect faces, align, extract features and match
faces1 = yunet.detect(img1)[1]
face1 = faces1[0][:-1] # take the first face and filter out score
faces2 = yunet.detect(img2)[1]
face2 = faces2[0][:-1]
aligned_face1 = sface.alignCrop(img1, face1)
aligned_face2 = sface.alignCrop(img2, face2)
feature1 = sface.feature(aligned_face1)
feature2 = sface.feature(aligned_face2)
score = sface.match(feature1, feature2)


#judgement whether in database

# load database from CSV file
database = {}
with open("database.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        try:
            name, feature = row[0], row[1:]
            database[name] = [float(x) for x in feature]
        except IndexError:
            print("Error: invalid row:", row)

# extract feature from image
image = cv.imread("3.png")
image = cv.resize(image, (480, 640))
image_copy = image.copy() # make a copy of the image to draw on
faces = yunet.detect(image)[1]
if len(faces) > 0:
    face = faces[0][:-1] # take the first face and filter out score
    aligned_face = sface.alignCrop(image, face)
    feature = sface.feature(aligned_face)
else:
    print("No faces detected in image.")

# check the shapes of the input arrays to make sure they are correct
if feature.shape != (1, 512):
    print("Error: feature array has invalid shape:", feature.shape)
else:
    # compare feature to features in database
    for name, db_feature in database.items():
        # convert database feature to numpy array
        db_feature_mat = np.array(db_feature, dtype=np.float32)
        # reshape array to match input shape of cv2.match()
        db_feature_mat = db_feature_mat.reshape(1, -1)
        score = sface.match(feature, db_feature_mat)
        if score > threshold:
            # draw bounding box and label on image copy
            x, y, w, h = face[0], face[1], face[2], face[3]
            cv.rectangle(image_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv.putText(image_copy, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            break
    else:
        print("Face not found in database.")

# display image with bounding box and label
cv.imshow("Matched Face", image_copy)
cv.waitKey(0)
cv.destroyAllWindows()















