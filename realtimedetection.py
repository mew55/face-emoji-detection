import cv2
from keras.models import model_from_json
import numpy as np

# Load model from JSON and weights
json_file = open("../emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("../emotiondetector.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Preprocessing function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Label mapping
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Start webcam
webcam = cv2.VideoCapture(0)

print("Press ESC to quit.")

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))               # Resize for speed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # Convert to grayscale

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Loop through detected faces
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))

        # Predict emotion
        img = extract_features(face_img)
        prediction = model.predict(img)
        predicted_label = labels[prediction.argmax()]

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

    # Show the output
    cv2.imshow("Output", frame)

    # Exit on pressing ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()

