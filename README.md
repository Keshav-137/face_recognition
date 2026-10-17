# Face Recognition using OpenCV

A simple face recognition project built using **Python and OpenCV**.

This project uses a webcam to collect face images, trains an **LBPH (Local Binary Patterns Histograms)** face recognizer, and then recognizes people in real time.

## Project Structure

```text
Face Recognition/
│
├── Data/
│   ├── 1.Keshav.1.jpg
│   ├── 1.Keshav.2.jpg
│   ├── 2.Sidh.1.jpg
│   └── ...
│
├── generate_dataset.py
├── train_classifier.py
├── detect_face_with_name.py
│
├── haarcascade_frontalface_default.xml
├── classifier.xml
├── id_to_name.json
│
└── README.md
```

## How the Project Works

The project has three main steps:

```text
Webcam
   ↓
Face Detection
   ↓
Save Face Images
   ↓
Train LBPH Model
   ↓
Live Face Recognition
   ↓
Display Name
```

### 1. Generate Dataset

`generate_dataset.py`

This script opens the webcam and detects the user's face using the Haar Cascade classifier.

The detected face is cropped, resized, and saved inside the `Data` folder.

Example:

```text
1.Keshav.1.jpg
1.Keshav.2.jpg
1.Keshav.3.jpg
```

The filename contains:

```text
ID.Name.ImageNumber
```

For example:

```text
1.Keshav.20.jpg
```

means:

* ID = 1
* Name = Keshav
* Image number = 20

---

### 2. Train the Model

`train_classifier.py`

This script reads the face images from the `Data` folder and trains the LBPH face recognizer.

It creates two important files:

```text
classifier.xml
id_to_name.json
```

`classifier.xml` contains the trained face recognition model.

`id_to_name.json` stores the relationship between the ID and person's name.

Example:

```json
{
    "1": "Keshav",
    "2": "Sidh"
}
```

---

### 3. Recognize Faces

`detect_face_with_name.py`

This script starts the webcam and performs real-time face recognition.

For every detected face, it:

1. Detects the face
2. Sends the face to the trained LBPH model
3. Gets the predicted ID
4. Finds the person's name from `id_to_name.json`
5. Displays the name on the webcam screen

If the face does not match a known person, it displays:

```text
UNKNOWN
```

## Technologies Used

* Python
* OpenCV
* Haar Cascade
* LBPH Face Recognizer
* NumPy
* PIL/Pillow
* JSON

## Requirements

Install the required Python packages:

```bash
pip install opencv-python opencv-contrib-python numpy pillow
```

> `opencv-contrib-python` is required because the LBPH face recognizer is part of OpenCV's contrib modules.

## How to Run

### Step 1: Collect Face Images

Run:

```bash
python generate_dataset.py
```

Enter the required user ID and name.

The captured images will be saved inside:

```text
Data/
```

### Step 2: Train the Model

Run:

```bash
python train_classifier.py
```

This will create:

```text
classifier.xml
id_to_name.json
```

### Step 3: Start Face Recognition

Run:

```bash
python detect_face_with_name.py
```

The webcam will open and the system will recognize trained faces.

## Important

The dataset path in `generate_dataset.py` may need to be changed depending on where the project is stored.

For example:

```python
E:/Programming/Python/Projects/Face Recognition/Data
```

Change it to the actual `Data` folder location on your computer.

## Limitations

This is a basic face recognition project and works best in controlled conditions.

Recognition can be affected by:

* Poor lighting
* Different face angles
* Low-quality camera
* Large changes in appearance
* Small training datasets

The project uses Haar Cascade and LBPH, which are traditional computer vision techniques.

## Future Improvements

Some possible improvements are:

* Use a modern deep-learning face detector
* Improve face recognition accuracy
* Add a proper GUI
* Store users in a database
* Add login/attendance functionality
* Add face registration from the application
* Improve unknown-face detection
* Replace hardcoded paths with relative paths

## Author

**Keshav Chandrakant Sonawane**

This project was created as a practical implementation of face detection and face recognition using Python and OpenCV.
