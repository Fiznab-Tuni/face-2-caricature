from __future__ import print_function

import copy

import cv2 as cv
import numpy as np
import scipy.io as sio
import math
import argparse

from os import listdir, walk
from os.path import isfile, join
import time
import traceback


def get_delay(start_time, fps=30):
    if (time.time() - start_time) > (1 / float(fps)):
        return 1
    else:
        return max(int((1 / float(fps)) * 1000 - (time.time() - start) * 1000), 1)


def add_sunglasses(img, p1, p2):
    print("Adding sunglasses")
    background = img
    h1, w1 = background.shape[:2]
    c1 = (int(w1/2), int(h1/2))
    dist = math.dist(p1, p2)
    delta = int(0.5*dist)
    print("points:", p1, p2)
    print("dist:", dist)

    overlay = cv.imread('../data/overlays/sunglasses001.png', cv.IMREAD_UNCHANGED)
    cv.imshow('Overlay original', overlay)

    srcTri = np.array([[0, 0], [overlay.shape[1] - 1, 0], [0, overlay.shape[0] - 1]]).astype(np.float32)
    dstTri = np.array([[p1[0]-delta, p1[1]-delta], [p2[0]+delta, p2[1]-delta],
                       [p1[0]-delta, p1[1]+delta]]).astype(np.float32)
    warp_mat = cv.getAffineTransform(srcTri, dstTri)
    overlay = cv.warpAffine(overlay, warp_mat, (w1, h1))
    cv.imshow('Overlay transformed', overlay)

    alpha_channel = overlay[:, :, 3] / 255
    overlay_colors = overlay[:, :, :3]

    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    h, w = overlay.shape[:2]
    background_subsection = background[0:h, 0:w]

    composite = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask
    background[0:h, 0:w] = composite

    return background


def update_map(ind, map_x, map_y):
    hx, wx = map_x.shape[:2]
    hy, wy = map_y.shape[:2]
    print("Remapping pixels..")
    print("Height: {}, Width: {}".format(hx, wx))

    ind = 4
    if ind == 0:
        for i in range(hx):
            for j in range(wx):
                if wx * 0.25 < j < wx * 0.75 and hx * 0.25 < i < hx * 0.75:
                    map_x[i,j] = 2 * (j - wx * 0.25) + 0.5
                    map_y[i,j] = 2 * (i - hy * 0.25) + 0.5
                else:
                    map_x[i,j] = 0
                    map_y[i,j] = 0

    elif ind == 1:
        for i in range(hx):
            map_x[i,:] = [x for x in range(wx)]
        for j in range(wy):
            map_y[:,j] = [hy-y for y in range(hy)]

    elif ind == 2:
        for i in range(hx):
            map_x[i,:] = [wx-x for x in range(wx)]
        for j in range(wy):
            map_y[:,j] = [y for y in range(hy)]

    elif ind == 3:
        for i in range(hx):
            map_x[i,:] = [wx-x for x in range(wx)]
        for j in range(wy):
            map_y[:,j] = [hy-y for y in range(hy)]

    elif ind == 4:
        simple_mode = True
        print("Using Pinch effect")
        print("In simple mode: {}".format(simple_mode))

        if simple_mode:
            if detected_nose:
                print("Using nose")
                center = (wx // 2, hx // 2)
                #center = detected_nose
                amount = -0.4
                angle = 1 * math.pi
            else:
                print("Using center")
                center = (wx // 2, hx // 2)
                amount = 0.7
                angle = 0 * math.pi

        else:
            center1 = (wx // 2, hx // 3)
            center2 = (wx // 2, int(hx // 1.4))
            amount1 = 2
            amount2 = -0.7
            angle = 0*math.pi

        if hx < wx:
            if simple_mode:
                radius = hx * 0.5
            else:
                radius = hx * 0.2
        else:
            if simple_mode:
                radius = wx * 0.5
            else:
                radius = wx * 0.2

        for i in range(hx):
            for j in range(wx):
                if simple_mode:
                    dist = math.dist(center, (j, i))
                else:
                    dist1 = math.dist(center1, (j, i))
                    dist2 = math.dist(center2, (j, i))
                    if dist1 < dist2:
                        center = center1
                        dist = dist1
                        amount = amount1
                    else:
                        center = center2
                        dist = dist2
                        amount = amount2

                if dist > radius or dist < 0.01:
                    map_x[i,j] = j
                    map_y[i,j] = i
                else:
                    d = dist/radius
                    trans = math.sin(math.pi*0.5*d) ** amount
                    dx = (j - center[0]) * trans
                    dy = (i - center[1]) * trans
                    e = (1 - d) ** 2
                    a = angle * e
                    c = math.cos(a)
                    s = math.sin(a)
                    map_x[i,j] = center[0] + c*dx - s*dy
                    map_y[i,j] = center[1] + s*dx + c*dy


def detect_smile(face, use_cutoff=True):
    h, w = face.shape[:2]

    if use_cutoff:
        print("Using cutoff for smiles")
        cutoff = int(0.7 * h)
        smileROI = face[cutoff:h, :]
        smiles = smile_cascade.detectMultiScale(smileROI)
    else:
        smiles = smile_cascade.detectMultiScale(face)

    #print("smiles:", len(smiles))
    if len(smiles) == 0:
        print("Using default value for smile")
        p1 = (int(0.3 * w), int(0.7 * h))
        p2 = (int(0.7 * w), int(0.9 * h))

    elif len(smiles) > 1:
        print("Non Maximum Suppression for smiles")
        scores = 0.9 * (np.ones(len(smiles)))
        score_threshold = 0.8
        nms_threshold = 0.05
        kept_indices = cv.dnn.NMSBoxes(smiles, scores, score_threshold, nms_threshold)
        # print(kept_indices)
        for i in kept_indices:
            (x2, y2, w2, h2) = smiles[i]
            if use_cutoff:
                p1 = (x2, y2 + cutoff)
                p2 = (x2 + w2, y2 + h2 + cutoff)
            else:
                p1 = (x2, y2)
                p2 = (x2 + w2, y2 + h2)

    else:
        for (x2, y2, w2, h2) in smiles:
            print("Detected smile with general detector")
            if use_cutoff:
                p1 = (x2, y2 + cutoff)
                p2 = (x2 + w2, y2 + h2 + cutoff)
            else:
                p1 = (x2, y2)
                p2 = (x2 + w2, y2 + h2)

    return p1, p2


def detect_nose(face, use_cutoff=True):
    h, w = face.shape[:2]

    if use_cutoff:
        print("Using cutoff for noses")
        cutoff = int(0.2 * h)
        noseROI = face[cutoff:h-cutoff, :]
        noses = nose_cascade.detectMultiScale(noseROI)
    else:
        noses = nose_cascade.detectMultiScale(face)

    #print("noses:", len(noses))
    if len(noses) == 0:
        print("Using default value for nose")
        p1 = (int(0.4 * w), int(0.4 * h))
        p2 = (int(0.6 * w), int(0.6 * h))

    elif len(noses) > 1:
        print("Non Maximum Suppression for noses")
        scores = 0.9 * (np.ones(len(noses)))
        score_threshold = 0.8
        nms_threshold = 0.05
        kept_indices = cv.dnn.NMSBoxes(noses, scores, score_threshold, nms_threshold)
        # print(kept_indices)
        for i in kept_indices:
            (x2, y2, w2, h2) = noses[i]
            if use_cutoff:
                p1 = (x2, y2 + cutoff)
                p2 = (x2 + w2, y2 + h2 + cutoff)
            else:
                p1 = (x2, y2)
                p2 = (x2 + w2, y2 + h2)

    else:
        for (x2, y2, w2, h2) in noses:
            print("Detected nose with general detector")
            if use_cutoff:
                p1 = (x2, y2 + cutoff)
                p2 = (x2 + w2, y2 + h2 + cutoff)
            else:
                p1 = (x2, y2)
                p2 = (x2 + w2, y2 + h2)

    return p1, p2


def detect_eyes(face, use_cutoff=True):
    h, w = face.shape[:2]
    no_detection = True
    radius = int(round(0.2 * w))
    points = []

    if use_cutoff:
        print("Using cutoff for eyes")
        cutoff = int(0.5 * h)
        eyeROI = face[:cutoff, :]
        eyes = eyes_cascade.detectMultiScale(eyeROI)
    else:
        eyeROI = face
        eyes = eyes_cascade.detectMultiScale(eyeROI)

    if len(eyes) == 2:
        print("Detected eyes with general detector")
        no_detection = False
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x2 + w2//2, y2 + h2//2)
            points.append(eye_center)

    else:
        right_eyes = right_eye_cascade.detectMultiScale(eyeROI)
        if len(right_eyes) == 2:
            print("Detected eyes with right eye detector")
            no_detection = False
            for (x2, y2, w2, h2) in right_eyes:
                eye_center = (x2 + w2//2, y2 + h2//2)
                points.append(eye_center)

        else:
            left_eyes = left_eye_cascade.detectMultiScale(eyeROI)
            if len(left_eyes) == 2:
                print("Detected eyes with left eye detector")
                no_detection = False
                for (x2, y2, w2, h2) in left_eyes:
                    eye_center = (x2 + w2//2, y2 + h2//2)
                    points.append(eye_center)

    if no_detection:
        print("Using default values for eyes")
        left_eye = (int(0.3 * w), int(0.4 * h))
        points.append(left_eye)
        right_eye = (int(0.7 * w), int(0.4 * h))
        points.append(right_eye)

    return points, radius


detected_nose = None
def detectAndDisplay(frame):
    global detected_nose
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    use_cutoff = True
    thickness = 2
    points = []
    ind = 0

    for (x,y,w,h) in faces:
        detected_nose = None

        # -- In each face, detect objects
        faceROI = frame_gray[y:y+h,x:x+w]

        # -- Detect smile
        p1, p2 = detect_smile(faceROI, use_cutoff)

        p1 = np.add(p1, ((x - w//3), (y - h//3)))
        p2 = np.add(p2, ((x + w//3), (y + h//3)))
        smileROI = frame[p1[1]:p2[1], p1[0]:p2[0]]

        map_x = np.zeros((smileROI.shape[0], smileROI.shape[1]), dtype=np.float32)
        map_y = np.zeros((smileROI.shape[0], smileROI.shape[1]), dtype=np.float32)

        update_map(ind, map_x, map_y)

        newROI = cv.remap(smileROI, map_x, map_y, cv.INTER_LINEAR)
        frame[p1[1]:p2[1], p1[0]:p2[0]] = newROI

        # -- Detect nose
        p1, p2 = detect_nose(faceROI, use_cutoff)

        width = p2[0] - p1[0]
        height = p2[1] - p1[1]
        detected_nose = (p1[0] + width//2, p1[1] + height//2)

        p1 = np.add(p1, ((x - w//4), (y - h//4)))
        p2 = np.add(p2, ((x + w//4), (y + h//4)))
        noseROI = frame[p1[1]:p2[1], p1[0]:p2[0]]

        map_x = np.zeros((noseROI.shape[0], noseROI.shape[1]), dtype=np.float32)
        map_y = np.zeros((noseROI.shape[0], noseROI.shape[1]), dtype=np.float32)

        update_map(ind, map_x, map_y)

        newROI = cv.remap(noseROI, map_x, map_y, cv.INTER_LINEAR)
        frame[p1[1]:p2[1], p1[0]:p2[0]] = newROI

        detected_nose = None

        # -- Detect eyes
        points, radius = detect_eyes(faceROI, use_cutoff)

        p1 = np.add(points[0], ((x - w//3), (y - h//3)))
        p2 = np.add(points[0], ((x + w//3), (y + h//3)))
        eyeROI = frame[p1[1]:p2[1], p1[0]:p2[0]]

        map_x = np.zeros((eyeROI.shape[0], eyeROI.shape[1]), dtype=np.float32)
        map_y = np.zeros((eyeROI.shape[0], eyeROI.shape[1]), dtype=np.float32)

        update_map(ind, map_x, map_y)

        newROI = cv.remap(eyeROI, map_x, map_y, cv.INTER_LINEAR)
        frame[p1[1]:p2[1], p1[0]:p2[0]] = newROI

        p1 = np.add(points[1], ((x - w//3), (y - h//3)))
        p2 = np.add(points[1], ((x + w//3), (y + h//3)))
        eyeROI = frame[p1[1]:p2[1], p1[0]:p2[0]]

        map_x = np.zeros((eyeROI.shape[0], eyeROI.shape[1]), dtype=np.float32)
        map_y = np.zeros((eyeROI.shape[0], eyeROI.shape[1]), dtype=np.float32)

        update_map(ind, map_x, map_y)

        newROI = cv.remap(eyeROI, map_x, map_y, cv.INTER_LINEAR)
        frame[p1[1]:p2[1], p1[0]:p2[0]] = newROI

    '''
    print("detector points", points)
    if len(points) == 2:
        #frame = add_sunglasses(frame, points[0], points[1])
        frame = add_sunglasses(frame, np.add(points[0], (x, y)), np.add(points[1], (x, y)))
    '''

    cv.imshow('Capture - Face detection', frame)
    cv.imwrite('../examples/' + jpeg, frame)


parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='../data/haarcascades/haarcascade_frontalface_alt.xml')
#parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='../data/haarcascades/haarcascade_eye.xml')
parser.add_argument('--left_eye_cascade', help='Path to left eye cascade.', default='../data/haarcascades/haarcascade_lefteye_2splits.xml')
parser.add_argument('--right_eye_cascade', help='Path to right eye cascade.', default='../data/haarcascades/haarcascade_righteye_2splits.xml')
parser.add_argument('--smile_cascade', help='Path to smile cascade.', default='../data/haarcascades/haarcascade_smile.xml')
parser.add_argument('--nose_cascade', help='Path to nose cascade.', default='../data/haarcascades/haarcascade_mcs_nose.xml')
#parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
parser.add_argument('--image', help='Path to input image.', default='../data/images/face001.png')
#parser.add_argument('--image', help='Path to input image.', default='../data/images/face004.jpg')
parser.add_argument('--images', help='Path to input images.', default='../data/images/')
parser.add_argument('--camera', help='Path to video.', default='../data/videos/visionface.avi')
parser.add_argument('--mat', help='path to widerface database (to be red)', default='../data/wider_face_split/wider_face_train.mat')
args = parser.parse_args()

face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
left_eye_cascade_name = args.left_eye_cascade
right_eye_cascade_name = args.right_eye_cascade
smile_cascade_name = args.smile_cascade
nose_cascade_name = args.nose_cascade

face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
left_eye_cascade = cv.CascadeClassifier()
right_eye_cascade = cv.CascadeClassifier()
smile_cascade = cv.CascadeClassifier()
nose_cascade = cv.CascadeClassifier()

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
    print('--(!)Error loading right eye cascade')
    exit(0)
if not smile_cascade.load(cv.samples.findFile(smile_cascade_name)):
    print('--(!)Error loading smile cascade')
    exit(0)
if not nose_cascade.load(cv.samples.findFile(nose_cascade_name)):
    print('--(!)Error loading nose cascade')
    exit(0)

#-- 2. Read one image
#image = cv.imread(cv.samples.findFile(args.image), cv.IMREAD_GRAYSCALE)
image = cv.imread(cv.samples.findFile(args.image))
if image is None:
    print('Could not open or find the image:', args.image)
    exit(0)

#-- 3. Load image directory
imagepath = args.images
#print(listdir(imagepath))
#exit(0)
files = []
for (dirpath, dirnames, filenames) in walk(imagepath):
    files.extend(filenames)
    break

"""
#-- 4. Load Widerface images from Matlab file
matfile = args.mat
subset = matfile.split('_')[-1].replace('.mat', '')
f = sio.loadmat(matfile)
if f is None:
    print('Could not open or find the mat-file:', args.mat)
    exit(0)

events = [f['event_list'][i][0][0] for i in range(len(f['event_list']))]
raw_files = [f['file_list'][i][0] for i in range(len(f['file_list']))]
#print(len(raw_files))
#raw_bbx = [f['face_bbx_list'][i][0] for i in range(len(f['face_bbx_list']))]

files = []
for group in raw_files:
    for file in group:
        filestr = file[0][0]
        #print(file, filestr)
        for ev in events:
            if filestr.startswith(ev.replace('--', '_')):
                files.append(str('../data/WIDER_%s/images/' % subset + ev + '/' + filestr + '.jpg'))
                break
"""

"""
#-- 5. Read the video stream
camera_device = args.camera
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
"""

"""
first_time = True

while True:
    start = time.time()
    
    jpeg = None
    
    if first_time:
        frame = image
        first_time = False

    else:
        ret, frame = cap.read()
"""

for jpeg in files:
    frame = cv.imread(imagepath + jpeg)
    print()
    print("Loading image:", jpeg)
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    detectAndDisplay(frame)

    # Limit FPS
    #if cv.waitKey(get_delay(start, fps=30)) & 0xFF == ord('q'):
    #    break  # q to quit

    # Wait for key press to update frame
    if cv.waitKey() & 0xFF == ord('q'):
        break  # q to quit

    #if cv.waitKey(10) == 27:
    #    break
