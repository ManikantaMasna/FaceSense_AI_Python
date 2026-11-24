# FaceSense AI - Python Edition

A desktop webcam application for face detection and facial-expression
classification using Python, OpenCV and DeepFace.

## Features

- Live webcam preview
- Face bounding boxes
- Seven expression prediction categories:
  happy, sad, angry, fearful, surprised, disgusted and neutral
- Model confidence scores
- Top three expression scores per detected face
- Screenshot capture
- Keyboard controls

## Requirements

- Windows, macOS or Linux
- Python 3.10 or 3.11 recommended
- Webcam
- Internet connection for initial package/model downloads

## Setup in VS Code

1. Extract the `FaceSense_AI_Python.zip` file.
2. Open the extracted `FaceSense_AI_Python` folder in VS Code.
3. Open Terminal -> New Terminal.
4. Create a virtual environment:

   Windows:
   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Upgrade pip:

   ```bash
   python -m pip install --upgrade pip
   ```

6. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

7. Run:

   ```bash
   python main.py
   ```

## Controls

- Q or ESC: quit
- S: save a screenshot in the `screenshots` folder

## Troubleshooting

### Camera does not open
- Close other apps using the webcam.
- Check OS camera permissions.
- Change `CAMERA_INDEX = 0` to `CAMERA_INDEX = 1` in `main.py` if you have multiple cameras.

### Installation issues
DeepFace uses TensorFlow and other dependencies. Use Python 3.10 or 3.11
and a fresh virtual environment. Installation can take several minutes.

### Slow performance
Change `ANALYZE_EVERY_N_FRAMES = 8` to `12` or `15` in `main.py`.
This reduces how often the AI analyzes frames.

## Privacy and limitations

The webcam frames are analyzed locally by this script and are not sent to
a server by the application code. Initial installation and model setup may
download software and model files from their package/model providers.

Expression outputs are model predictions from visible facial patterns.
They do not reliably establish a person's actual feelings, intentions,
personality or mental state. Accuracy varies with lighting, camera angle,
occlusion and other factors. Do not use this as a high-stakes assessment.
