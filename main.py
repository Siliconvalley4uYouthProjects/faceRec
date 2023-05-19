from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from google.cloud import vision
from google.oauth2.service_account import Credentials
from flask_cors import CORS
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'
cors = CORS(app)
SCOPES = ['https://www.googleapis.com/auth/classroom.courses.readonly']
credentials = Credentials.from_service_account_file('facialdetection-384720-04866abf3cf6.json')
client = vision.ImageAnnotatorClient(credentials=credentials)

socketio = SocketIO(app, cors_allowed_origins='*')

# Raspberry Pi camera setup
camera = cv2.VideoCapture(0)
camera.set(3, 640)  # Set width
camera.set(4, 480)  # Set height


@app.route('/')
def index():
    return render_template('webcam.html')


@socketio.on('connect')
def test_connect():
    print('Client connected')


@socketio.on('video_feed')
def video_feed():
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        _, buffer = cv2.imencode('.jpg', frame)
        binary_data = buffer.tobytes()

        image = vision.Image(content=binary_data)
        response = client.face_detection(image=image)
        faces = response.face_annotations
        dominant_emotion = "no emotion"

        # Analyze facial expressions for each detected face
        for face in faces:
            emotions = {}
            if face.joy_likelihood != vision.Likelihood.UNKNOWN:
                emotions['joy'] = face.joy_likelihood
            if face.sorrow_likelihood != vision.Likelihood.UNKNOWN:
                emotions['sorrow'] = face.sorrow_likelihood
            if face.anger_likelihood != vision.Likelihood.UNKNOWN:
                emotions['anger'] = face.anger_likelihood
            if face.surprise_likelihood != vision.Likelihood.UNKNOWN:
                emotions['surprise'] = face.surprise_likelihood
            if emotions:
                # Determine dominant emotion
                dominant_emotion = max(emotions, key=emotions.get)
                print(dominant_emotion)
        
        # Send the emotion result to the frontend
        emit('emotion_result', {'result': dominant_emotion})

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    socketio.run(app)