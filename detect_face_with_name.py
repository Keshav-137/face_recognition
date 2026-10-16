import cv2
import numpy as np
import json

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf, id_to_name):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

    for (x, y, w, h) in features:
        id, pred = clf.predict(gray_image[y:y + h, x:x + w])
        confidence = int(100 * (1 - pred / 300))

        if confidence > 70:
            name = id_to_name.get(str(id), "Unknown")  # JSON object keys are strings
            cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 0, 255], 1, cv2.LINE_AA)

    return img

def recognize(img, clf, faceCascade, id_to_name):
    img = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), clf, id_to_name)
    return img

# Load the Haar Cascade for face detection
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the trained face recognizer
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

# Load the ID to name mapping
with open("id_to_name.json", "r") as f:
    id_to_name = json.load(f)

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, img = video_capture.read()
    if not ret:
        break
    img = recognize(img, clf, faceCascade, id_to_name)
    cv2.imshow("Face Detection", img)

    # Exit on pressing Enter key
    if cv2.waitKey(1) == 13:
        break

# Release the video capture object and close windows
video_capture.release()
cv2.destroyAllWindows()