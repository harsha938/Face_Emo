import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from collections import deque, Counter

# Load the pre-trained emotion recognition model
model = load_model('emotion_model.h5')

# Define the emotions that the model can predict (must match the order used during training)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

# Create a queue for smoothing predictions (using the last N frames)
smoothing_queue_size = 10  # Number of frames to consider for smoother emotion prediction
emotion_queues = {}

# A dictionary to store the cumulative count of each emotion detected
emotion_statistics = Counter()

# Load the face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up real-time plotting for emotion statistics
plt.ion()  # Turn on interactive mode to update the plot in real-time
fig, ax = plt.subplots()
emotion_names = list(emotion_labels)
bar_width = 0.35
bar_positions = np.arange(len(emotion_names))

# Create an initial bar chart to visualize emotion distribution
bars = ax.bar(bar_positions, [0] * len(emotion_names), bar_width, align='center')
ax.set_xticks(bar_positions)
ax.set_xticklabels(emotion_names)
ax.set_ylim(0, 100)

# Start the loop to continuously analyze frames from the webcam
while True:
    # Capture the current frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to grayscale (model was trained on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        # Extract the face region and resize it to the model's expected input size
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_resized = face_resized / 255.0  # Normalize pixel values to [0, 1]
        face_resized = face_resized.reshape(1, 48, 48, 1)  # Reshape to match input dimensions

        # Predict the emotion based on the face image
        prediction = model.predict(face_resized)
        emotion_index = np.argmax(prediction)  # Get the index of the highest confidence
        predicted_emotion = emotion_labels[emotion_index]  # Map index to emotion label
        confidence = prediction[0][emotion_index] * 100  # Get confidence percentage

        # Add the predicted emotion to the smoothing queue for this face
        if i not in emotion_queues:
            emotion_queues[i] = deque(maxlen=smoothing_queue_size)
        emotion_queues[i].append(predicted_emotion)

        # Apply smoothing by choosing the most common emotion from the queue
        smoothed_emotion = Counter(emotion_queues[i]).most_common(1)[0][0]

        # Update the cumulative count of the detected emotion
        emotion_statistics[smoothed_emotion] += 1

        # Draw a rectangle around the detected face and display the predicted emotion and confidence
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
        cv2.putText(frame, f"{smoothed_emotion} ({confidence:.1f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Update the bar chart with real-time emotion statistics
    emotion_counts = [emotion_statistics[emotion] for emotion in emotion_labels]
    for i, bar in enumerate(bars):
        bar.set_height(emotion_counts[i])  # Update the height of each bar

    # Redraw the plot with updated values to reflect current emotion statistics
    plt.draw()
    plt.pause(0.01)

    # Display the current frame with the rectangle and emotion label
    cv2.imshow('Real-Time Emotion Detection', frame)

    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close the matplotlib plot window
plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot with cumulative emotion statistics
