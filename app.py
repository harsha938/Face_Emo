import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2

# Load FER2013 dataset
def load_data(train_dir, img_size=(48, 48)):
    """
    Load images from a directory and convert to numpy arrays
    """
    labels = []
    images = []

    # Loop through each emotion category in the dataset
    for emotion in os.listdir(train_dir):
        emotion_folder = os.path.join(train_dir, emotion)
        if os.path.isdir(emotion_folder):
            for image_name in os.listdir(emotion_folder):
                image_path = os.path.join(emotion_folder, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image = cv2.resize(image, img_size)
                    images.append(image)
                    labels.append(emotion)
    
    # Convert the images and labels into numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Normalize the images to [0, 1] range
    images = images / 255.0

    return images, labels

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Convert emotion labels to numeric values
def encode_labels(labels):
    # Convert dataset labels to lowercase and match with emotion_labels (case insensitive)
    labels = [label.lower() for label in labels]  # Convert dataset labels to lowercase
    encoded_labels = [emotion_labels.index(label.capitalize()) for label in labels]  # Convert emotion to capitalize
    return np.array(encoded_labels)

# Load the dataset
train_dir = "D:/ML/Face_Emo/train"  # Adjust to your dataset path
images, labels = load_data(train_dir)

# Encode labels into one-hot vectors
labels = encode_labels(labels)
labels = to_categorical(labels, num_classes=len(emotion_labels))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reshape the data to (samples, height, width, channels) as expected by CNNs
X_train = X_train.reshape(-1, 48, 48, 1)
X_val = X_val.reshape(-1, 48, 48, 1)

# Define the CNN model
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Layer
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output for the dense layer
model.add(Flatten())

# Dense layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 emotions

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val))

# Save the trained model
model.save('emotion_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {test_acc:.4f}")
