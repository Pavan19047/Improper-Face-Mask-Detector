# Face Mask Detection System

This project implements a real-time face mask detection system using deep learning and computer vision techniques.

## Features

- **Real-time Detection**: Uses webcam to detect faces and classify mask usage
- **Deep Learning Model**: Custom CNN architecture trained on face mask dataset
- **Multiple Classes**: Detects 3 categories:
  - `with_mask` - Person wearing mask correctly
  - `without_mask` - Person not wearing mask
  - `mask_weared_incorrect` - Person wearing mask incorrectly

## Setup Instructions

### 1. Environment Setup
The project has been configured with a Python virtual environment and all required dependencies:

- TensorFlow/Keras for deep learning
- OpenCV for computer vision
- NumPy, Pandas for data processing
- Matplotlib, Seaborn for visualization
- Scikit-learn for machine learning utilities

### 2. Dataset
The project uses the Face Mask Detection dataset from Kaggle with:
- **Total Images**: 853 images
- **Processed Faces**: 4,072 face crops (80x80 pixels)
- **Class Distribution**:
  - with_mask: 3,232 images
  - without_mask: 717 images
  - mask_weared_incorrect: 123 images

### 3. Model Architecture
- **Input**: 80x80x3 RGB images
- **Layers**: 
  - 2 Convolutional blocks (Conv2D + MaxPool + Dropout)
  - Dense layers with LeakyReLU activation
  - Softmax output for 3-class classification
- **Parameters**: ~2.3M trainable parameters

### 4. Training Results
- **Validation Accuracy**: ~93%
- **Training Time**: ~9 epochs (with early stopping)
- **Optimizer**: Adam with learning rate 0.001

## Running the Application

### Option 1: Jupyter Notebook (Recommended)
1. Open `DL Face Detection Final Code.ipynb`
2. All cells have been executed and the model is trained
3. Run the last cell with `cv2.VideoCapture(0)` for real-time detection
4. Press 'q' to quit the camera feed

### Option 2: Python Demo Script
```bash
# Run the demo script
python face_mask_demo.py
```
Or double-click `run_demo.bat` on Windows

### Option 3: Manual Execution
```bash
# Activate virtual environment
.venv\Scripts\activate

# Run specific components
python -c "from face_mask_demo import main; main()"
```

## File Structure
```
├── DL Face Detection Final Code.ipynb  # Main notebook with complete implementation
├── face_mask_demo.py                   # Standalone demo script
├── run_demo.bat                        # Windows batch file to run demo
├── dataset/                            # Sample test images
│   ├── face.png
│   ├── mask.jpg
│   └── no_mask_face.jpg
├── haarcascades/                       # Haar cascade classifiers
│   └── haarcascade_frontalface_alt2.xml
└── kaggle/input/face-mask-detection/   # Training dataset
    ├── images/                         # Original images
    └── annotations/                    # XML annotation files
```

## Model Performance
- **Face Detection**: Haar Cascade classifier for fast face detection
- **Mask Classification**: Custom CNN with 93% validation accuracy
- **Real-time Performance**: ~30-40ms per prediction
- **Precision Metrics**: 
  - Micro: High precision across all classes
  - Macro: Balanced performance with slight bias toward majority class

## Usage Tips
1. **Lighting**: Ensure good lighting for better face detection
2. **Distance**: Stay 2-3 feet from camera for optimal detection
3. **Angle**: Face the camera directly for best results
4. **Masks**: System can detect surgical masks, cloth masks, and improper usage

## Technical Details
- **Face Detection**: OpenCV Haar Cascade
- **Image Preprocessing**: Resize to 80x80, normalize pixel values
- **Data Augmentation**: Rotation, shift, zoom for training robustness
- **Framework**: TensorFlow 2.x with Keras API

## Troubleshooting
- **Camera not working**: Check camera permissions and ensure no other apps are using it
- **Low accuracy**: Ensure good lighting and face visibility
- **Performance issues**: Consider reducing image processing resolution

## Next Steps
- Implement model saving/loading for faster startup
- Add support for multiple face detection
- Enhance with mask type classification (surgical, N95, cloth)
- Deploy as web application or mobile app