# 🎯 Face Emoji Detection

This is a real-time facial emotion detection system built using Convolutional Neural Networks (CNN), Keras, and OpenCV. The model classifies facial expressions into seven categories: *Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise*, using webcam input.

---

## 🛠️ Tools & Technologies Used

- Python  
- Keras (TensorFlow backend)  
- OpenCV  
- NumPy  

---

## 📁 Files Included

- main.py – Entry point of the application  
- realtimedetection.py – Real-time webcam-based emotion detection  
- trainmodel.ipynb – Notebook for training the emotion detection model  
- emotiondetector.json – Saved model architecture (JSON format)  
- requirements.txt – List of dependencies  

---

## 🧠 Model Training Details

- Trained over *30 epochs*  
- Achieved approximately *60–65% accuracy*  
- Trained on a labeled dataset of facial emotions (custom organized)

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/mew55/face-emoji-detection.git
   cd face-emoji-detection

## Install required packages:
pip install -r requirements.txt

## Run the real-time emotion detection:
python realtimedetection.py

📷 Output
When you run the app, your webcam will turn on. It will detect faces and display the predicted emotion label above the face in real-time.

📌 Note
Due to GitHub file size limits, the trained model file (emotiondetector.h5) and image dataset are not included here. You can request them via [Google Drive link or email].

🌟 Feel free to star this repo if you liked it!

Author: Mihir Umrotkar
