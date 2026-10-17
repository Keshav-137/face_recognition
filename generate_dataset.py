import cv2
import os
import time
from config import DATA_DIR
from config import FACE_IMAGE_SIZE, SAMPLES_PER_POSE, CAPTURE_INTERVAL_SECONDS


def face_cropped(img, face_classifier, mirrored=False):
    detection_img = cv2.flip(img, 1) if mirrored else img
    gray = cv2.cvtColor(detection_img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    if mirrored:
        x = img.shape[1] - x - w
    return img[y:y + h, x:x + w], (x, y, w, h)


def generate_dataset():
    cascade_dir = cv2.data.haarcascades
    frontal_classifier = cv2.CascadeClassifier(
        os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
    )
    profile_classifier = cv2.CascadeClassifier(
        os.path.join(cascade_dir, "haarcascade_profileface.xml")
    )
    if frontal_classifier.empty() or profile_classifier.empty():
        raise RuntimeError("Could not load the frontal and profile face cascades.")

    user_id = input("Enter User ID: ")
    user_name = input("Enter User Name: ")

    os.makedirs(DATA_DIR, exist_ok=True)
    img_id = 0
    for filename in os.listdir(DATA_DIR):
        parts = os.path.splitext(filename)[0].split(".")
        if (
            len(parts) >= 3
            and parts[0] == user_id
            and ".".join(parts[1:-1]) == user_name
            and parts[-1].isdigit()
        ):
            img_id = max(img_id, int(parts[-1]))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    poses = (
        ("front", frontal_classifier, False, "Look straight at the camera."),
        ("left profile", profile_classifier, False, "Turn your face to your left."),
        ("right profile", profile_classifier, True, "Turn your face to your right."),
    )
    collected_count = 0
    stopped_early = False
    try:
        for pose_name, classifier, mirrored, instruction in poses:
            print(f"Collecting {pose_name} images. {instruction}")
            captured_for_pose = 0
            next_capture_at = time.monotonic()

            while captured_for_pose < SAMPLES_PER_POSE:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to capture a frame from the webcam.")

                detection = face_cropped(frame, classifier, mirrored)
                if detection is not None:
                    cropped_face, (x, y, w, h) = detection
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    now = time.monotonic()
                    if now >= next_capture_at:
                        face = cv2.resize(cropped_face, FACE_IMAGE_SIZE)
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        next_id = img_id + 1
                        file_name_path = os.path.join(
                            DATA_DIR, f"{user_id}.{user_name}.{next_id}.jpg"
                        )
                        if not cv2.imwrite(file_name_path, face):
                            raise RuntimeError(f"Failed to save sample to '{file_name_path}'.")
                        img_id = next_id
                        captured_for_pose += 1
                        collected_count += 1
                        next_capture_at = now + CAPTURE_INTERVAL_SECONDS

                cv2.putText(
                    frame,
                    f"{pose_name}: {captured_for_pose}/{SAMPLES_PER_POSE}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{instruction} Press Enter or Q to stop.",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Dataset Capture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 13):
                    stopped_early = True
                    break

            if stopped_early:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Collected {collected_count} images in this session.")
    if not stopped_early and collected_count == SAMPLES_PER_POSE * len(poses):
        print("Dataset collection is complete (100 front, 100 left profile, 100 right profile).")


if __name__ == "__main__":
    generate_dataset()
