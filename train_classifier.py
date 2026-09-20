import numpy as np
from PIL import Image
import os
import cv2
import json

def train_classifier(data_dir):
    if not os.path.exists(data_dir):
        print(f"Error: The directory '{data_dir}' does not exist. Training failed.")
        return False  # Indicate failure

    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg") or f.endswith(".png")]
    
    if len(path) == 0:
        print("No images found in the given directory. Training failed.")
        return False  # Indicate failure
    
    faces = []
    ids = []
    id_to_name = {}  # Dictionary to map id to name
    
    successful_images = 0

    for image in path:
        try:
            img = Image.open(image).convert('L')
            imageNp = np.array(img, 'uint8')
            filename = os.path.split(image)[1]
            id = int(filename.split(".")[0])  # Extract ID from filename (first part)
            name = filename.split(".")[1]  # Extract name from filename (second part)
            
            faces.append(imageNp)
            ids.append(id)
            id_to_name[id] = name  # Store the name with the ID
            successful_images += 1
        except FileNotFoundError as e:
            print(f"Error: File not found {image}: {e}")
        except Image.UnidentifiedImageError as e:
            print(f"Error: Could not open or read image {image}: {e}")
        except ValueError as e:
            print(f"Error: Invalid ID in filename {image}: {e}")
        except Exception as e:
            print(f"Error processing image {image}: {e}")

    ids = np.array(ids)

    if not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        print("Error: LBPHFaceRecognizer_create is not available in your OpenCV version. Training failed.")
        return False  # Indicate failure

    if successful_images < 2:
        print("Error: Insufficient successful images to train the classifier.  Need at least 2. Training failed.")
        return False # Indicate failure
        
    try:
        clf = cv2.face.LBPHFaceRecognizer_create()
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