import numpy as np
import os
import cv2
import json
from config import DATA_DIR
from config import FACE_IMAGE_SIZE


def parse_image_identity(filename):
    parts = os.path.splitext(os.path.basename(filename))[0].split(".")
    if len(parts) < 3 or not parts[-1].isdigit():
        raise ValueError("expected a name, ID, and numeric sample number")

    if parts[0].isdigit():
        user_id = int(parts[0])
        name = ".".join(parts[1:-1])
    elif parts[-2].isdigit():
        name = ".".join(parts[:-2])
        user_id = int(parts[-2])
    else:
        raise ValueError("could not find a numeric ID in the filename")

    if not name:
        raise ValueError("name is missing from the filename")
    return user_id, name


def train_classifier(data_dir=DATA_DIR):
    if not os.path.exists(data_dir):
        print(f"Error: The directory '{data_dir}' does not exist. Training failed.")
        return False  # Indicate failure

    path = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".jpg", ".png"))
    )
    
    if len(path) == 0:
        print("No images found in the given directory. Training failed.")
        return False  # Indicate failure
    
    faces = []
    ids = []
    id_to_name = {}  # Dictionary to map id to name
    id_name_conflicts = {}
    
    successful_images = 0

    for image in path:
        try:
            imageNp = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if imageNp is None:
                print(f"Error: Could not open or read image {image}.")
                continue
            imageNp = cv2.resize(imageNp, FACE_IMAGE_SIZE)
            user_id, name = parse_image_identity(image)
            
            faces.append(imageNp)
            ids.append(user_id)
            id_key = str(user_id)
            existing_name = id_to_name.setdefault(id_key, name)
            if existing_name != name:
                id_name_conflicts.setdefault(id_key, {existing_name}).add(name)
            successful_images += 1
        except FileNotFoundError as e:
            print(f"Error: File not found {image}: {e}")
        except ValueError as e:
            print(f"Error: Invalid ID in filename {image}: {e}")
        except Exception as e:
            print(f"Error processing image {image}: {e}")

    ids = np.array(ids)

    if id_name_conflicts:
        conflicts = "; ".join(
            f"ID {user_id}: {', '.join(sorted(names))}"
            for user_id, names in sorted(id_name_conflicts.items())
        )
        print(f"Error: Each ID must belong to one person. Conflicting dataset names: {conflicts}. Training failed.")
        return False

    face_module = getattr(cv2, "face", None)
    if face_module is None or not hasattr(face_module, "LBPHFaceRecognizer_create"):
        print("Error: LBPHFaceRecognizer_create is unavailable. Install opencv-contrib-python. Training failed.")
        return False  # Indicate failure

    if successful_images < 2:
        print("Error: Insufficient successful images to train the classifier.  Need at least 2. Training failed.")
        return False # Indicate failure
        
    try:
        clf = face_module.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("classifier.xml")

        # Save the mapping of IDs to names
        with open("id_to_name.json", "w") as f:
            json.dump(id_to_name, f)

        print("Training complete. Classifier saved as 'classifier.xml'. Training succeeded.")
        return True # Indicate success

    except Exception as e:
        print(f"Error during training: {e}. Training failed.")
        return False # Indicate failure


if __name__ == "__main__":
    train_classifier()
