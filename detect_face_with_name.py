import cv2
import json
import os
from config import FACE_IMAGE_SIZE, RECOGNITION_CONFIDENCE_THRESHOLD


def detect_faces(gray_image, face_cascade, profile_cascade):
    faces = list(face_cascade.detectMultiScale(gray_image, 1.1, 10))
    faces.extend(profile_cascade.detectMultiScale(gray_image, 1.1, 5))

    flipped_gray = cv2.flip(gray_image, 1)
    for x, y, w, h in profile_cascade.detectMultiScale(flipped_gray, 1.1, 5):
        faces.append((gray_image.shape[1] - x - w, y, w, h))

    unique_faces = []
    for face in sorted(faces, key=lambda box: box[2] * box[3], reverse=True):
        x, y, w, h = face
        area = w * h
        duplicate = False
        for other_x, other_y, other_w, other_h in unique_faces:
            intersection_w = max(0, min(x + w, other_x + other_w) - max(x, other_x))
            intersection_h = max(0, min(y + h, other_y + other_h) - max(y, other_y))
            if intersection_w * intersection_h / area > 0.5:
                duplicate = True
                break
        if not duplicate:
            unique_faces.append(face)
    return unique_faces


def draw_boundary(img, classifier, profile_classifier, color, clf, id_to_name):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = detect_faces(gray_image, classifier, profile_classifier)

    for (x, y, w, h) in features:
        face = cv2.resize(gray_image[y:y + h, x:x + w], FACE_IMAGE_SIZE)
        id, pred = clf.predict(face)
        confidence = int(100 * (1 - pred / 300))

        if confidence > RECOGNITION_CONFIDENCE_THRESHOLD:
            name = id_to_name.get(str(id), "Unknown")  # JSON object keys are strings
            label = f"{name} {confidence}%"
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            label = f"UNKNOWN {confidence}%"
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 0, 255], 1, cv2.LINE_AA)

    return img


def recognize(img, clf, faceCascade, profileCascade, id_to_name):
    img = draw_boundary(
        img, faceCascade, profileCascade, (255, 255, 255), clf, id_to_name
    )
    return img


def main():
    cascade_dir = cv2.data.haarcascades
    faceCascade = cv2.CascadeClassifier(
        os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
    )
    profileCascade = cv2.CascadeClassifier(
        os.path.join(cascade_dir, "haarcascade_profileface.xml")
    )
    if faceCascade.empty() or profileCascade.empty():
        raise RuntimeError("Could not load the frontal and profile face cascades.")

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    with open("id_to_name.json", "r") as f:
        id_to_name = json.load(f)

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise RuntimeError("Could not open the webcam.")
    try:
        while True:
            ret, img = video_capture.read()
            if not ret:
                raise RuntimeError("Failed to capture a frame from the webcam.")
            img = recognize(img, clf, faceCascade, profileCascade, id_to_name)
            cv2.imshow("Face Detection", img)

            if cv2.waitKey(1) == 13:
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()