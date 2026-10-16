import cv2
import os
from config import DATA_DIR


def face_cropped(img, face_classifier):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    return [img[y:y + h, x:x + w] for (x, y, w, h) in faces]


def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    user_id = input("Enter User ID: ")
    user_name = input("Enter User Name: ")

    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    img_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        cropped_faces = face_cropped(frame, face_classifier)
        for cropped_face in cropped_faces:
            img_id += 1
            face = cv2.resize(cropped_face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = os.path.join(DATA_DIR, f"{user_id}.{user_name}.{img_id}.jpg")
            if not cv2.imwrite(file_name_path, face):
                print(f"Failed to save sample to '{file_name_path}'.")
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Cropped Face", face)

            if img_id >= 200:
                break

        if cv2.waitKey(1) == 13 or img_id >= 200:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed.")


if __name__ == "__main__":
    generate_dataset()
