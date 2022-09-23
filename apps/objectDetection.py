from __future__ import print_function
import cv2 as cv
import argparse

import time
import traceback


def get_delay(start_time, fps=30):
    if (time.time() - start_time) > (1 / float(fps)):
        return 1
    else:
        return max(int((1 / float(fps)) * 1000 - (time.time() - start) * 1000), 1)


def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    thickness = 2
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), thickness)

        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect objects
        smiles = smile_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in smiles:
            # eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            p1 = (x + x2, y + y2)
            p2 = (x + x2 + w2, y + y2 + h2)
            # radius = int(round((w2 + h2) * 0.15))
            # frame = cv.circle(frame, eye_center, radius, (0, 0, 255), 4)
            frame = cv.rectangle(frame, p1, p2, (0, 0, 255), thickness)

        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), thickness)

        left_eyes = left_eye_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in left_eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.20))
            frame = cv.circle(frame, eye_center, radius, (255, 255, 0), thickness)

        right_eyes = right_eye_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in right_eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.15))
            frame = cv.circle(frame, eye_center, radius, (0, 255, 255), thickness)

    cv.imshow('Capture - Face detection', frame)


parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='../data/haarcascades/haarcascade_frontalface_alt.xml')
#parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='../data/haarcascades/haarcascade_eye.xml')
parser.add_argument('--left_eye_cascade', help='Path to eyes cascade.', default='../data/haarcascades/haarcascade_lefteye_2splits.xml')
parser.add_argument('--right_eye_cascade', help='Path to eyes cascade.', default='../data/haarcascades/haarcascade_righteye_2splits.xml')
parser.add_argument('--smile_cascade', help='Path to eyes cascade.', default='../data/haarcascades/haarcascade_smile.xml')
#parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
parser.add_argument('--image', help='Path to input image.', default='../data/images/face001.png')
parser.add_argument('--camera', help='Path to video.', default='../data/videos/visionface.avi')
args = parser.parse_args()

face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
left_eye_cascade_name = args.left_eye_cascade
right_eye_cascade_name = args.right_eye_cascade
smile_cascade_name = args.smile_cascade

face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
left_eye_cascade = cv.CascadeClassifier()
right_eye_cascade = cv.CascadeClassifier()
smile_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
if not left_eye_cascade.load(cv.samples.findFile(left_eye_cascade_name)):
    print('--(!)Error loading left eye cascade')
    exit(0)
if not right_eye_cascade.load(cv.samples.findFile(right_eye_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
if not smile_cascade.load(cv.samples.findFile(smile_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

#image = cv.imread(cv.samples.findFile(args.image), cv.IMREAD_GRAYSCALE)
image = cv.imread(cv.samples.findFile(args.image))
if image is None:
    print('Could not open or find the image:', args.image)
    exit(0)

camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

first_time = True

while True:
    start = time.time()

    if first_time:
        frame = image
        first_time = False

    else:
        ret, frame = cap.read()

    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    detectAndDisplay(frame)

    # Limit FPS
    if cv.waitKey(get_delay(start, fps=30)) & 0xFF == ord('q'):
        break  # q to quit

    # Wait for key press to update frame
    if cv.waitKey() & 0xFF == ord('q'):
        break  # q to quit

    #if cv.waitKey(10) == 27:
    #    break
