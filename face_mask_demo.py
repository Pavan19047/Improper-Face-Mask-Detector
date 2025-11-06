#!/usr/bin/env python3
"""
Face Mask Detection Demo Script
This script demonstrates the face mask detection system on sample images.
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, BatchNormalization, MaxPooling2D, LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import xml.etree.ElementTree as ET
from tqdm import tqdm

def create_model():
    """Create and return the CNN model architecture"""
    model = Sequential()
    
    model.add(Conv2D(32,(3,3),padding='SAME',activation='relu',input_shape=(80,80,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(64,(3,3),padding='SAME',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(90,activation='relu'))
    model.add(Dropout(0.5))
    model.add(LeakyReLU(negative_slope=0.05))
    model.add(Dense(3,activation = "softmax"))
    
    return model

def get_box(obj):
    """Extract bounding box coordinates from XML annotation"""
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    return [xmin, ymin, xmax, ymax]

def load_and_prepare_data():
    """Load and prepare the training data"""
    print("Loading and preparing data...")
    
    # Set up paths
    images_path = './kaggle/input/face-mask-detection/images'
    annotation_path = './kaggle/input/face-mask-detection/annotations'
    
    Image_width = 80
    Image_height = 80
    Image_array = []
    Labels = []
    
    # Process annotations and images
    for file in tqdm(sorted(os.listdir(annotation_path)), desc='Preparing data...'):
        file_path = annotation_path + "/" + file
        xml = ET.parse(file_path)
        root = xml.getroot()
        image_path = images_path + "/" + root[1].text
        
        for bndbox in root.iter('bndbox'):
            [xmin, ymin, xmax, ymax] = get_box(bndbox)
            img = cv2.imread(image_path)
            face_img = img[ymin:ymax,xmin:xmax]
            face_img = cv2.resize(face_img,(Image_width,Image_height))
            Image_array.append(np.array(face_img))
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            Labels.append(name)  # Store as string, not numpy array
    
    # Normalize and encode data
    X = np.array(Image_array)
    X = X/255
    
    le = LabelEncoder()
    y = le.fit_transform(Labels)
    y = to_categorical(y, 3)
    
    return X, y, Labels

def train_model(X, y):
    """Train the face mask detection model"""
    print("Training model...")
    
    model = create_model()
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Quick training for demo (reduce epochs for faster demo)
    history = model.fit(X, y, 
                        batch_size=64,
                        epochs=5,  # Reduced for demo
                        validation_split=0.2,
                        verbose=1)
    
    return model

def detect_faces_and_predict(image_path, model):
    """Detect faces in image and predict mask status"""
    # Load Haar Cascade
    face_model = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_model.detectMultiScale(gray,
                                        scaleFactor=1.1, 
                                        minNeighbors=5, 
                                        minSize=(60,60),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    labels = ['incorrect', 'with_mask', 'without_mask']
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face region
        face_img = img_rgb[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (80, 80))
        face_img = face_img.reshape(-1, 80, 80, 3)
        face_img = face_img / 255.0
        
        # Make prediction
        pred = np.argmax(model.predict(face_img, verbose=0), axis=-1)
        prediction_text = labels[pred[0]]
        
        # Draw bounding box and label
        color = (0, 255, 0) if pred[0] == 1 else (255, 0, 0)  # Green for mask, red for no mask
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_rgb, prediction_text, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return img_rgb

def main():
    """Main function to run the face mask detection demo"""
    print("Face Mask Detection Demo")
    print("========================")
    
    # Check if we have the required data
    if not os.path.exists('./kaggle/input/face-mask-detection'):
        print("Error: Dataset not found. Please ensure the dataset is in the correct location.")
        return
    
    try:
        # Load and prepare data
        X, y, Labels = load_and_prepare_data()
        print(f"Loaded {len(X)} images with {len(set(Labels))} classes")
        
        # Train model
        model = train_model(X, y)
        print("Model training completed!")
        
        # Test on sample images from dataset folder
        sample_images = [
            './dataset/face.png',
            './dataset/mask.jpg',
            './dataset/no_mask_face.jpg'
        ]
        
        # Create figure for displaying results
        fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 5))
        if len(sample_images) == 1:
            axes = [axes]
        
        for i, img_path in enumerate(sample_images):
            if os.path.exists(img_path):
                print(f"Processing {img_path}...")
                result_img = detect_faces_and_predict(img_path, model)
                
                if result_img is not None:
                    axes[i].imshow(result_img)
                    axes[i].set_title(f"Detection Result: {os.path.basename(img_path)}")
                    axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, f"Could not load\n{os.path.basename(img_path)}", 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f"Image not found:\n{os.path.basename(img_path)}", 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('./face_mask_detection_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\nDemo completed! Results saved as 'face_mask_detection_results.png'")
        print("\nTo use with webcam, run the notebook cell with cv2.VideoCapture(0)")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please make sure all required dependencies are installed and data is available.")

if __name__ == "__main__":
    main()