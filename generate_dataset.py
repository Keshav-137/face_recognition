import cv2
import os

def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None

        for (x, y, w, h) in faces:
            return img[y:y + h, x:x + w]

    user_id = input("Enter User ID: ")
    user_name = input("Enter User Name: ")

    data_dir = "E:/Programming/Python/Projects/Face Recognition/Data"
    os.makedirs(data_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    img_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        cropped_face = face_cropped(frame)
        if cropped_face is not None:
            img_id += 1
            face = cv2.resize(cropped_face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = f"{data_dir}/{user_id}.{user_name}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Cropped Face", face)

            if cv2.waitKey(1) == 13 or img_id == 200:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed.")

generate_dataset()