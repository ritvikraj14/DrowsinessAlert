#!/usr/bin/python
# -*- coding: utf-8 -*-
from twilio.rest import Client
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

import geocoder
g = geocoder.ip('me')
loc1 = g.latlng
loc2 = ['{:.2f}'.format(x) for x in loc1]
print g.latlng


def call():
    account_sid = 'ACb9f3d225b2e27b63265fd1f184bd2e3d'
    auth_token = '16a84a67a9f4abb1bef00a5269d6c55c'
    client = Client(account_sid, auth_token)

    call = client.calls.create(to='+917060334053', from_='+13232759475'
                               ,
                               url='http://demo.twilio.com/docs/voice.xml'
                               )

    print call.sid


def message():
    account_sid = 'ACb9f3d225b2e27b63265fd1f184bd2e3d'
    auth_token = '16a84a67a9f4abb1bef00a5269d6c55c'
    client = Client(account_sid, auth_token)
    message = client.messages.create(to='+917060334053',
            from_='+13232759475',
            body='Accident has occurred and gps location is'
            + ', '.join(loc2))
    print message.sid


def sound_alarm(path):
    playsound.playsound(path)


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[11])
    B = dist.euclidean(mouth[2], mouth[10])

    C = dist.euclidean(mouth[3], mouth[9])
    D = dist.euclidean(mouth[4], mouth[8])
    E = dist.euclidean(mouth[5], mouth[7])

    F = dist.euclidean(mouth[0], mouth[6])

    # compute the eye aspect ratio

    ear = (B + C + D) / (2.0 * F)

    return ear


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio

    ear = (A + B) / (2.0 * C)

    return ear


ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required=True,
                help='path to facial landmark predictor')
ap.add_argument('-a', '--alarm', type=str, default='',
                help='path alarm .WAV file')
ap.add_argument('-w', '--webcam', type=int, default=0,
                help='index of webcam on system')
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48
EYE_AR_ACC = 150

MOUTH_AR_THRESH = 0.60

# MOUTH_AR_CONSEC_FRAMES = 36

COUNTER = 0
COUNTER1 = 0
ALARM_ON = False

print '[INFO] loading facial landmark predictor...'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

(mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

# start the video stream thread

print '[INFO] starting video stream thread...'
vs = VideoStream(src=args['webcam']).start()
time.sleep(1.0)  # use to warm up camera

# loop over frames from the video stream

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame

    rects = detector(gray, 0)

    # loop over the face detections

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        mouth1 = shape[mouth_start:mouth_end]
        MAR = mouth_aspect_ratio(mouth1)

        mouth_hull = cv2.convexHull(mouth1)
        cv2.drawContours(frame, [mouth_hull], -1, (0, 0xFF, 0), 1)

        # average the eye aspect ratio together for both eyes

        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0xFF, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0xFF, 0), 1)

        if ear < EYE_AR_THRESH or MAR > MOUTH_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:

                # if the alarm is not on then turn it on

                if not ALARM_ON:
                    ALARM_ON = True

                    if args['alarm'] != '':
                        t = Thread(target=sound_alarm,
                                   args=(args['alarm'], ))
                        t.deamon = True
                        t.start()

                # draw an alarm on the frame

                cv2.putText(
                    frame,
                    'DROWSINESS ALERT!',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0xFF),
                    2,
                    )

    if ear < EYE_AR_THRESH:
        COUNTER1 += 1
        if COUNTER1 >= EYE_AR_ACC:
            message()
            call()
    else:

        COUNTER = 0
        ALARM_ON = False

    cv2.putText(
        frame,
        'EAR: {:.2f}'.format(ear),
        (300, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0xFF),
        2,
        )
    cv2.putText(
        frame,
        'MAR: {:.2f}'.format(MAR),
        (300, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0xFF),
        2,
        )

    # show the frame

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop

    if key == ord('q'):
        break

# do a bit of cleanup

cv2.destroyAllWindows()
vs.stop()
