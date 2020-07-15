import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
from flask import Flask, Response
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from PIL import Image, ImageFont, ImageDraw
from datetime import datetime
import numpy as np
import emoji
import dlib
import os
import smtplib
import ssl
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO
from smbus2 import SMBus
from mlx90614 import MLX90614

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mask_detected = False
body_temperature = None
temp_high = False
temp_low = False
not_proceed = False
proceed = False
visitor_id = 0
promt = None


class Servo:
    def __init__(self, pin):
        self.pin = pin

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)

        self.servo = GPIO.PWM(self.pin, 50)
        self.servo.start(2.5)  # Initialization

    def setAngle(self, angle):
        duty = angle / 18 + 2
        GPIO.output(self.pin, True)
        self.servo.ChangeDutyCycle(duty)
        time.sleep(0.2)
        GPIO.output(self.pin, False)
        self.servo.ChangeDutyCycle(0)

    def reset(self):
        GPIO.output(self.pin, True)
        self.servo.ChangeDutyCycle(2.5)
        time.sleep(1)
        GPIO.output(self.pin, False)
        self.servo.ChangeDutyCycle(0)

    def stop(self):
        self.servo.stop()


class PID:
    def __init__(self, kP=1, kI=0, kD=0):
        # initialize gains
        self.kP = kP
        self.kI = kI
        self.kD = kD

    def initialize(self):
        # intialize the current and previous time
        self.currTime = time.time()
        self.prevTime = self.currTime

        # initialize the previous error
        self.prevError = 0

        # initialize the term result variables
        self.cP = 0
        self.cI = 0
        self.cD = 0

    def update(self, error, sleep=0.2):
        # pause for a bit
        time.sleep(sleep)

        # grab the current time and calculate delta time
        self.currTime = time.time()
        deltaTime = self.currTime - self.prevTime

        # delta error
        deltaError = error - self.prevError

        # proportional term
        self.cP = error

        # integral term
        self.cI += error * deltaTime

        # derivative term and prevent divide by zero
        self.cD = (deltaError / deltaTime) if deltaTime > 0 else 0

        # save previous time and error for the next update
        self.prevTime = self.currTime
        self.prevError = error

        # sum the terms and return
        return sum([
            self.kP * self.cP,
            self.kI * self.cI,
            self.kD * self.cD])

    def reset(self):
        self.currTime = time.time()
        self.prevTime = self.currTime

        # initialize the previous error
        self.prevError = 0

        # initialize the term result variables
        self.cP = 0
        self.cI = 0
        self.cD = 0


class VideoCamera(object):
    def __init__(self):

        self.servoxPIN = 32
        self.servoyPIN = 33

        self.camera = PiCamera()
        self.camera.framerate = 32
        self.camera.rotation = 180
        self.rawCapture = PiRGBArray(self.camera)
        # allow the camera to warmup
        time.sleep(0.1)

        self.send_email = os.environ.get('SEND_EMAIL')

        self.mid = 0

        self.temp = None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

        self.model = load_model('models/mask_detector.h5')

        self.mask_detection_completed = False
        
        self.mask_count = 0

        self.temperature_check_completed = False

        self.operation_status_failed = False

        if self.send_email == 'TRUE':
            self.sender_email = os.environ.get('EMAIL_ID')
            self.receiver_email = os.environ.get('EMAIL_ID')
            self.password = os.environ.get('EMAIL_PWD')

            self.message = MIMEMultipart("alternative")
            self.message["Subject"] = "Alert: A New Person Entered the Premises"
            self.message["From"] = self.sender_email
            self.message["To"] = self.receiver_email

        self.servoX = Servo(self.servoxPIN)
        self.servoY = Servo(self.servoyPIN)

        self.servoX.setAngle(90)
        self.servoY.setAngle(90)

        self.bus = SMBus(1)
        self.sensor = MLX90614(self.bus, address=0x5A)

    def detect_mask(self, image):
        copy_img = image.copy()

        resized = cv2.resize(copy_img, (254, 254))

        resized = img_to_array(resized)
        resized = preprocess_input(resized)

        resized = np.expand_dims(resized, axis=0)

        mask, _ = self.model.predict([resized])[0]

        return mask

    def email(self, img_path, temp, mask):
        with open(img_path, 'rb') as f:
            # set attachment mime and file name, the image type is png
            mime = MIMEBase('image', 'png', filename='img1.png')
            # add required header data:
            mime.add_header('Content-Disposition', 'attachment', filename='img1.png')
            mime.add_header('X-Attachment-Id', '0')
            mime.add_header('Content-ID', '<0>')
            # read attachment file content into the MIMEBase object
            mime.set_payload(f.read())
            # encode with base64
            encoders.encode_base64(mime)
            # add MIMEBase object to MIMEMultipart object
            self.message.attach(mime)

        body = MIMEText('''
        <html>
            <body>
                <h1>Alert</h1>
                <h2>A new has Person entered the Premises</h2>
                <h2>Body Temperature: {}</h2>
                <h2>Mask: {}</h2>
                <h2>Time: {}</h2>
                <p>
                    <img src="cid:0">
                </p>
            </body>
        </html>'''.format(temp, mask, datetime.now()), 'html', 'utf-8')

        self.message.attach(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(self.sender_email, self.password)
            server.sendmail(
                self.sender_email, self.receiver_email, self.message.as_string()
            )

    def __del__(self):
        self.servoX.stop()
        self.servoY.stop()

        self.GPIO.cleanup()

        self.bus.close()

    def get_frame(self):
        success, img = True, self.frame.array

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        im_p = Image.fromarray(img)
        draw = ImageDraw.Draw(im_p)
        font_3 = ImageFont.truetype("Arial Unicode.ttf", 100)

        global mask_detected, body_temperature, temp_high, temp_low, not_proceed, proceed, visitor_id, promt

        promt = None

        if success:

            if self.mask_detection_completed is False:
                mask_prob = self.detect_mask(img)

                if mask_prob > 0.5:
                    self.mask_count += 1

                    if self.mask_count >= 5:
                        self.mask_detection_completed = True
                        self.pidX = PID()
                        self.pidY = PID()

                        self.pidX.initialize()
                        self.pidY.initialize()

                elif mask_prob < 0.5:
                    mask_detected = False

            elif self.mask_detection_completed:
                if self.temperature_check_completed is False:
                    mask_detected = True

                    faces = self.detector(img_gray, 0)

                    if len(faces) > 0:

                        for face in faces:

                            landmarks = self.predictor(img_gray, face)

                            im_n = np.array(im_p)

                            landmarks_list = []
                            for i in range(0, landmarks.num_parts):
                                landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

                                cv2.circle(im_n, (landmarks.part(i).x, landmarks.part(i).y), 4, (255, 255, 255), -1)

                            dist = np.sqrt((landmarks.part(21).x - landmarks.part(22).x) ** 2 + (
                                    landmarks.part(21).y - landmarks.part(22).y) ** 2)

                            face_ptx, face_pty = (int((landmarks.part(21).x + landmarks.part(22).x) / 2),
                                                  int((landmarks.part(21).y + landmarks.part(22).y) / 2) - int(dist))

                            cv2.circle(im_n, (face_ptx, face_pty), 4, (0, 255, 0), -1)

                            Y, X, _ = img.shape

                            sensor_ptx, sensor_pty = (int(X / 2), int(Y / 3))

                            cv2.rectangle(im_n, (sensor_ptx - 10, sensor_pty - 10), (sensor_ptx + 10, sensor_pty + 10),
                                          (255, 0, 0),
                                          4)
                            cv2.circle(im_n, (sensor_ptx, sensor_pty), 5, (255, 0, 0), -1)

                            diff_x, diff_y = sensor_ptx - face_ptx, sensor_pty - face_pty

                            if -10 < diff_x < 10 and -10 < diff_y < 10:
                                self.temp = self.sensor.get_amb_temp()
                                body_temperature = self.temp
                                if self.temp > 100:
                                    temp_high = True
                                    self.operation_status_failed = True
                                    self.temperature_check_completed = True
                                else:
                                    temp_low = True

                                    self.temperature_check_completed = True
                            else:
                                im_p = Image.fromarray(im_n)
                                draw = ImageDraw.Draw(im_p)
                                servoX.setAngle(-1 * self.pidX.update(diff_x))
                                servoY.setAngle(-1 * self.pidY.update(diff_y))
                                if diff_x > 0:
                                    draw.text((700, 500), '→', (0, 0, 0), font=font_3)
                                elif diff_x < 0:
                                    draw.text((700, 500), '←', (0, 0, 0), font=font_3)

                                if diff_y > 0:
                                    draw.text((600, 500), '↓', (0, 0, 0), font=font_3)
                                elif diff_y < 0:
                                    draw.text((600, 500), '↑', (0, 0, 0), font=font_3)
                    else:
                        promt = 'No Face Detected Please remove mask'

                elif self.temperature_check_completed:
                    proceed = True
                    c_id = os.environ.get('COUNTER_ID')
                    cv2.imwrite('pictures/{}.jpg'.format(str(c_id)), img)
                    self.email('pictures/{}.jpg'.format(c_id), self.temp, 'Wearing')
                    os.environ['COUNTER_ID'] = str(int(c_id) + 1)
                    self.mid += 1

                    # Reset
                    self.mask_detection_completed = False
                    self.mask_count = 0
                    self.temperature_check_completed = False
                    self.temp = None
                    mask_detected = False
                    body_temperature = None
                    temp_high = False
                    temp_low = False
                    not_proceed = False
                    proceed = False
                    promt = None

                elif self.operation_status_failed:
                    self.mask_detection_completed = False
                    self.mask_count = 0
                    self.temperature_check_completed = False
                    self.temp = None
                    self.operation_status_failed = False
                    mask_detected = False
                    body_temperature = None
                    temp_high = False
                    temp_low = False
                    not_proceed = False
                    proceed = False
                    promt = None

            visitor_id = self.mid + 1
            img = np.array(im_p)
            self.rawCapture.truncate(0)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


server = Flask(__name__)
app = dash.Dash(__name__, server=server)

app.title = 'TouchFree'


@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


app.layout = html.Div(id='outer-div', children=[
    html.Div(id='banner-div', children=[
        html.Div([
            html.H1('TouchFree',
                    style={'display': 'inline-block', 'margin-top': '0px'}),
            html.H2('AI based Mask Detection and Temperature Check', style={'margin-top': '0px'})
        ], style={'text-align': 'center'})
    ], style={'padding': '10px'}),
    html.Div(id='control-container', children=[
        html.Div(id='video-container', className='pretty_container', children=[
            html.Img(id='video-feed', src='/video_feed', height=450, width=850)
        ], style={'width': '60%', 'height': '455px', 'display': 'inline-block'}),
        html.Div(id='data-container', className='pretty_container', style={'width': '30%',
                                                                           'height': '455px',
                                                                           'display': 'inline-block',
                                                                           'text-align': 'center',
                                                                           'vertical-align': 'top',
                                                                           'margin': 'center'}),
        dcc.Interval(id='counter-interval', interval=100, n_intervals=0)
    ]),
    html.Div(id='proceed-container', className='pretty_container', children=[], style={'width': '100%',
                                                                                       'height': '90px',
                                                                                       'text-align': 'center',
                                                                                       'padding': '0px'})
])


@app.callback(Output('data-container', 'children'),
              [Input('counter-interval', 'n_intervals')])
def labels_updater(n):

    if promt is not None:
        promt_label = html.H2(promt)
    else:
        promt_label = html.H1(' ')

    if mask_detected:
        mask_label = html.H2('Mask Detected', style={'color': 'green'})
        if body_temperature is not None:
            if temp_high:
                msg_label = html.H2('High Body Temperature Detected', style={'color': 'red'})
                temp_label = temp_label = daq.Thermometer(
                    id='thermometer',
                    min=95,
                    max=105,
                    value=body_temperature,
                    color='red',
                    style={'padding': '10px'}
                )
            elif temp_low:
                msg_label = html.H2('Normal Body Temperature', style={'color': 'green'})
                temp_label = temp_label = daq.Thermometer(
                    id='thermometer',
                    min=95,
                    max=105,
                    value=body_temperature,
                    color='green'
                )
        else:
            temp_label = daq.Thermometer(
                id='thermometer',
                min=95,
                max=105,
                value=body_temperature,
                color='blue'
            )
            msg_label = html.H2('Checking Temperature...')
    else:
        mask_label = html.H2('No Mask Detected', style={'color': 'red'})
        temp_label = temp_label = daq.Thermometer(
                id='thermometer',
                min=95,
                max=105,
                value=0,
                color='blue'
            )
        msg_label = html.H2('Detecting Mask...')
    return [html.H2(f'Visitor Id: {visitor_id}'),
            mask_label,
            temp_label,
            promt_label,
            msg_label]


@app.callback(Output('proceed-container', 'children'),
              [Input('counter-interval', 'n_intervals')])
def proceed_updater(n):
    proceed_label = html.H1('Please Stand Still!')
    if proceed:
        proceed_label = html.H1('Please Proceed', style={'color': 'green'})
    elif not_proceed:
        proceed_label = html.H1('You cannot Proceed!', style={'color': 'red'})
    elif not mask_detected:
        proceed_label = html.H1('You cannot Proceed!', style={'color': 'red'})
    elif mask_detected:
        proceed_label = html.H1('Please Align the green dot with blue dot!', style={'color': 'blue'})

    return proceed_label


if __name__ == '__main__':
    app.run_server()
