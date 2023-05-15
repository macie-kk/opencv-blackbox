from flask import Flask, render_template, Response
import cv2
import time
from collections import deque
import numpy as np
import urllib.request
# import mediapipe as mp


app = Flask(__name__)
video_url = f"http://{ip_address}:{port}/video"
stream = urllib.request.urlopen(video_url)

byte_buffer = bytes()

save_last_mins = 5
save_next_secs = 10
fps = 30

last_minutes = deque(maxlen=save_last_mins * 60 * fps)
next_seconds = deque(maxlen=save_next_secs * fps)

# load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# initialize pose estimator
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process the frame for pose detection
    # pose_results = pose.process(frame_rgb)

    # # draw skeleton on the frame
    # mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


def gen_frames():
    global last_minutes, next_seconds

    while True:
        # Read a chunk of data from the video stream
        chunk = stream.read(1024)

        # Append the chunk to the byte buffer
        byte_buffer += chunk

        # Find the start and end markers of a frame
        start_marker = byte_buffer.find(b'\xff\xd8')
        end_marker = byte_buffer.find(b'\xff\xd9')

        # If both markers are found, decode and display the frame
        if start_marker != -1 and end_marker != -1:
            frame_data = byte_buffer[start_marker:end_marker + 2]
            byte_buffer = byte_buffer[end_marker + 2:]

            # Convert the frame data to a NumPy array
            frame = np.frombuffer(frame_data, dtype=np.uint8)

            # Decode the frame as an image
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            detect(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # add frame to last_minutes queue
            last_minutes.append((time.time(), frame))

            # add frame in next ten seconds queue
            next_seconds.append((time.time(), frame))

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/save_buffer', methods=['POST'])
def save_buffer():
    global last_minutes, next_seconds

    last = last_minutes.copy()

    while True:
        filename = f'buffer_{time.time()}.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 30, (640, 480))
        start_time = time.time()

        for timestamp, frame in last:
            img = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
            out.write(img)

        while next_seconds[0][0] < start_time:
            time.sleep(1/30)

        next = next_seconds.copy()

        for timestamp, frame in next:
            img = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
            out.write(img)

        out.release()
        return 'Buffer saved!'


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
