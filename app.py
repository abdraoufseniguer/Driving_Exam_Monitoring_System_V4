from flask import Flask, render_template, Response
import cv2
import numpy as np
import datetime
import os

app = Flask(__name__)

# Initialize the camera
camera_index = 0
camera = cv2.VideoCapture(camera_index)

if not camera.isOpened():
    raise RuntimeError(f"Error: Could not open camera at index {camera_index}")

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

known_face_encodings = []
known_face_names = []
current_ids = {}
next_id = 1

def generate_frames():
    global next_id, current_ids
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            new_ids = set()
            for (x, y, w, h) in faces:
                face_encoding = gray_frame[y:y+h, x:x+w]
                face_id = None
                for known_face_encoding, known_face_id in zip(known_face_encodings, known_face_names):
                    if np.array_equal(face_encoding, known_face_encoding):
                        face_id = known_face_id
                        break
                if face_id is None:
                    known_face_encodings.append(face_encoding)
                    face_id = next_id
                    known_face_names.append(next_id)
                    next_id += 1
                new_ids.add(face_id)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {face_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            for face_id in current_ids - new_ids:
                log_event(face_id, "left")

            for face_id in new_ids - current_ids:
                log_event(face_id, "entered")

            current_ids = new_ids

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def log_event(face_id, event):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{now} - ID: {face_id} {event}"
    with open("logs.txt", "a") as log_file:
        log_file.write(log_entry + "\n")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0,port="5000")
