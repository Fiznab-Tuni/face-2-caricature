from __future__ import print_function

import cv2 as cv
import dlib
import numpy as np
import scipy.io as sio

import argparse
import math
from os import listdir, walk
from os.path import isfile, join
import random
import time


def get_delay(start_time, fps=30):
    # limit frames per second
    if (time.time() - start_time) > (1 / float(fps)):
        return 1
    else:
        return max(int((1 / float(fps)) * 1000 - (time.time() - start) * 1000), 1)


def rect_to_bb(rect):
    # take a dlib bounding and convert it to the OpenCV format (x, y, w, h)
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return x, y, w, h


def bb_to_rect(x, y, w, h):
    # take a OpenCV bounding (x, y, w, h) and convert it to the dlib format
    left = x
    top = y
    right = w + x
    bottom = h + y

    return dlib.rectangle(left, top, right, bottom)


def shape_to_np(shape, dtype="int"):
    # initialize numpy array of (x, y)-coordinates
    parts = shape.num_parts
    coords = np.zeros((parts, 2), dtype=dtype)

    # loop over the facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for i in range(parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def add_sunglasses(img, p1, p2):
    # add sunglasses over an image of a face
    print("Adding sunglasses")
    background = img
    h1, w1 = background.shape[:2]
    dist = math.dist(p1, p2)
    delta = int(dist*0.5)

    # overlay image is initialized when starting the program
    overlay = np.copy(overlay_original)
    #cv.imshow('Overlay original', overlay)

    # make affine transformation from source points to the destination points
    source_points = np.array([[0, 0], [overlay.shape[1] - 1, 0], [0, overlay.shape[0] - 1]]).astype(np.float32)
    destination_points = np.array([[p1[0]-delta, p1[1]-delta], [p2[0]+delta, p2[1]-delta],
                                   [p1[0]-delta, p1[1]+delta]]).astype(np.float32)
    warp_mat = cv.getAffineTransform(source_points, destination_points)
    overlay = cv.warpAffine(overlay, warp_mat, (w1, h1))
    #cv.imshow('Overlay transformed', overlay)

    # place overlay over the background
    alpha_channel = overlay[:, :, 3] / 255
    overlay_colors = overlay[:, :, :3]
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    h2, w2 = overlay.shape[:2]
    background_subsection = background[0:h2, 0:w2]

    composite = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask
    background[0:h2, 0:w2] = composite

    return background


def update_map(ind, map_x, map_y):
    # update pixel map for image filtering
    hx, wx = map_x.shape[:2]
    hy, wy = map_y.shape[:2]
    print("Remapping pixels..")
    print("Height: {}, Width: {}".format(hx, wx))

    # shrink image
    if ind == 0:
        for i in range(hx):
            for j in range(wx):
                if wx * 0.25 < j < wx * 0.75 and hx * 0.25 < i < hx * 0.75:
                    map_x[i,j] = 2 * (j - wx * 0.25) + 0.5
                    map_y[i,j] = 2 * (i - hy * 0.25) + 0.5
                else:
                    map_x[i,j] = 0
                    map_y[i,j] = 0

    # flip image vertically
    elif ind == 1:
        for i in range(hx):
            map_x[i,:] = [x for x in range(wx)]
        for j in range(wy):
            map_y[:,j] = [hy-y for y in range(hy)]

    # flip image horizontally
    elif ind == 2:
        for i in range(hx):
            map_x[i,:] = [wx-x for x in range(wx)]
        for j in range(wy):
            map_y[:,j] = [y for y in range(hy)]

    # flip image horizontally and vertically
    elif ind == 3:
        for i in range(hx):
            map_x[i,:] = [wx-x for x in range(wx)]
        for j in range(wy):
            map_y[:,j] = [hy-y for y in range(hy)]


def update_map_pinch(map_x, map_y, amount=0.6, angle=0.0, pinch=False):
    # update pixel map for pinch/bulge effect
    hx, wx = map_x.shape[:2]
    center = (wx // 2, hx // 2)
    #print("Remapping pixels..")
    #print("Height: {}, Width: {}".format(hx, wx))

    if pinch:
        #print("Using Pinch effect")
        amount = -amount
    else:
        #print("Using Bulge effect")
        amount = amount

    if hx < wx:
        radius = hx * 0.5
    else:
        radius = wx * 0.5

    for i in range(hx):
        for j in range(wx):
            dist = math.dist(center, (j, i))
            if dist > radius or dist < 0.01:
                map_x[i, j] = j
                map_y[i, j] = i
            else:
                d = dist / radius
                trans = math.sin(math.pi * 0.5 * d) ** amount
                dx = (j - center[0]) * trans
                dy = (i - center[1]) * trans
                e = (1 - d) ** 2
                a = angle * e
                c = math.cos(a)
                s = math.sin(a)
                map_x[i, j] = center[0] + c*dx - s*dy
                map_y[i, j] = center[1] + s*dx + c*dy


def filter_smile(frame, shape, amount=0.6, angle=0.0, pinch=False):
    # filter mouth using facial landmarks
    #center = ((shape[62][0] + shape[66][0]) // 2, (shape[62][1] + shape[66][1]) // 2)
    center = ((shape[48][0] + shape[54][0]) // 2, (shape[62][1] + shape[66][1]) // 2)
    #delta = int(math.dist(center, shape[30]))
    delta = int(math.dist(center, shape[48]))

    p1 = np.add(center, (-delta, -delta))
    p2 = np.add(center, (delta, delta))
    region = frame[p1[1]:p2[1], p1[0]:p2[0]]

    map_x = np.zeros((region.shape[0], region.shape[1]), dtype=np.float32)
    map_y = np.zeros((region.shape[0], region.shape[1]), dtype=np.float32)
    update_map_pinch(map_x, map_y, amount=amount, angle=angle, pinch=pinch)

    filtered_region = cv.remap(region, map_x, map_y, cv.INTER_LINEAR)
    frame[p1[1]:p2[1], p1[0]:p2[0]] = filtered_region

    #cv.rectangle(frame, p1, p2, (0, 255, 0), 2)

    return frame


def filter_nose(frame, shape, amount=0.6, angle=0.0, pinch=True):
    # filter nose using facial landmarks
    #center = ((shape[30][0] + shape[33][0]) // 2, (shape[30][1] + shape[33][1]) // 2)
    center = ((shape[31][0] + shape[35][0]) // 2, (shape[30][1] + shape[33][1]) // 2)
    delta = 2 * (int(math.dist(shape[30], shape[33])))

    p1 = np.add(center, (-delta, -delta))
    p2 = np.add(center, (delta, delta))
    region = frame[p1[1]:p2[1], p1[0]:p2[0]]

    map_x = np.zeros((region.shape[0], region.shape[1]), dtype=np.float32)
    map_y = np.zeros((region.shape[0], region.shape[1]), dtype=np.float32)
    update_map_pinch(map_x, map_y, amount=amount, angle=angle, pinch=pinch)

    filtered_region = cv.remap(region, map_x, map_y, cv.INTER_LINEAR)
    frame[p1[1]:p2[1], p1[0]:p2[0]] = filtered_region

    #cv.rectangle(frame, p1, p2, (0, 255, 0), 2)

    return frame


def filter_eye(frame, shape, left, amount=0.6, angle=0.0, pinch=False):
    # filter eye using facial landmarks
    if left:
        #center = ((shape[37][0] + shape[38][0] + shape[40][0] + shape[41][0]) // 4,
        #          (shape[37][1] + shape[38][1] + shape[40][1] + shape[41][1]) // 4)
        center = ((shape[36][0] + shape[39][0]) // 2,
                  (shape[37][1] + shape[38][1] + shape[40][1] + shape[41][1]) // 4)
    else:
        #center = ((shape[43][0] + shape[44][0] + shape[46][0] + shape[47][0]) // 4,
        #          (shape[43][1] + shape[44][1] + shape[46][1] + shape[47][1]) // 4)
        center = ((shape[42][0] + shape[45][0]) // 2,
                  (shape[43][1] + shape[44][1] + shape[46][1] + shape[47][1]) // 4)

    delta = int(math.dist(center, shape[27]))
    p1 = np.add(center, (-delta, -delta))
    p2 = np.add(center, (delta, delta))
    region = frame[p1[1]:p2[1], p1[0]:p2[0]]

    map_x = np.zeros((region.shape[0], region.shape[1]), dtype=np.float32)
    map_y = np.zeros((region.shape[0], region.shape[1]), dtype=np.float32)
    update_map_pinch(map_x, map_y, amount=amount, angle=angle, pinch=pinch)

    filtered_region = cv.remap(region, map_x, map_y, cv.INTER_LINEAR)
    frame[p1[1]:p2[1], p1[0]:p2[0]] = filtered_region

    #cv.rectangle(frame, p1, p2, (0, 255, 0), 2)

    return frame


def detect_smile(face, use_cutoff=True):
    # detect mouth in a face using Haar cascade detector
    h, w = face.shape[:2]

    if use_cutoff:
        print("Using cutoff for smiles")
        cutoff = int(0.7 * h)
        smileROI = face[cutoff:h, :]
        smiles = smile_cascade.detectMultiScale(smileROI)
    else:
        smiles = smile_cascade.detectMultiScale(face)

    if len(smiles) == 0:
        print("Using default value for smile")
        p1 = (int(0.3 * w), int(0.7 * h))
        p2 = (int(0.7 * w), int(0.9 * h))

    elif len(smiles) > 1:
        print("Non Maximum Suppression for smiles")
        # using default values for confidence scores
        scores = 0.9 * (np.ones(len(smiles)))
        score_threshold = 0.8
        nms_threshold = 0.05
        kept_indices = cv.dnn.NMSBoxes(smiles, scores, score_threshold, nms_threshold)
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
    # detect nose in a face using Haar cascade detector
    h, w = face.shape[:2]

    if use_cutoff:
        print("Using cutoff for noses")
        cutoff = int(0.2 * h)
        noseROI = face[cutoff:h-cutoff, :]
        noses = nose_cascade.detectMultiScale(noseROI)
    else:
        noses = nose_cascade.detectMultiScale(face)

    if len(noses) == 0:
        print("Using default value for nose")
        p1 = (int(0.4 * w), int(0.4 * h))
        p2 = (int(0.6 * w), int(0.6 * h))

    elif len(noses) > 1:
        print("Non Maximum Suppression for noses")
        # using default values for confidence scores
        scores = 0.9 * (np.ones(len(noses)))
        score_threshold = 0.8
        nms_threshold = 0.05
        kept_indices = cv.dnn.NMSBoxes(noses, scores, score_threshold, nms_threshold)
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
    # detect eyes in a face using Haar cascade detectors
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


class FilterPreset:
    # preset values for pinch/bulge gain and whirl angle
    def __init__(self, index=0):
        if index == 1:
            print("Using preset: old timer")
            self.amount = -0.75
            angle = 0
            self.nose_angle = 0
            self.smile_angle = 0
            self.left_angle = 0
            self.right_angle = 0
        elif index == 2:
            print("Using preset: alien")
            self.amount = 0.75
            angle = 0
            self.nose_angle = 0
            self.smile_angle = 0
            self.left_angle = 0
            self.right_angle = 0
        elif index == 3:
            print("Using preset: twister")
            self.amount = 0.25
            angle = 0.5 * math.pi
            self.nose_angle = angle
            self.smile_angle = angle
            self.left_angle = -angle
            self.right_angle = angle
        elif index == 4:
            print("Using preset: angry")
            self.amount = -0.25
            angle = 0.5 * math.pi
            self.nose_angle = 0
            self.smile_angle = angle
            self.left_angle = -angle
            self.right_angle = angle
        elif index == 5:
            print("Using preset: sleepy")
            self.amount = 0
            angle = 0.5 * math.pi
            self.nose_angle = 0
            self.smile_angle = 0
            self.left_angle = angle
            self.right_angle = -angle
        else:
            print("Using random values")
            self.amount = random.uniform(-1.0, 1.0)
            self.nose_angle = random.uniform(-0.5, 0.5) * math.pi
            self.smile_angle = random.uniform(-0.5, 0.5) * math.pi
            self.left_angle = random.uniform(-0.5, 0.5) * math.pi
            self.right_angle = random.uniform(-0.5, 0.5) * math.pi


def detect_and_display(frame, index=0):
    # detect faces in an image and process them
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # detect faces in a gray scale image
    faces = face_cascade.detectMultiScale(frame_gray)
    #print("Faces:", len(faces))
    #rects = face_detector(frame_gray, 1)
    #print("Faces:", len(rects))
    #h, w = frame_gray.shape[:2]
    #print(w, h)

    # load preset values for pinch/bulge gain and whirl angle
    preset = FilterPreset(index)
    '''
    #amount = -0.8
    #amount = random.uniform(-1.0, 1.0)
    amount = -1.0 + (index*(1/4))
    #angle = -0.5*math.pi
    #angle = 0.0 * math.pi
    bound = 0.5
    #twist = True
    '''

    # loop over the face detections
    #for (i, rect) in enumerate(rects):
    for (x, y, w, h) in faces:
        # determine the facial landmarks for the face region
        rect = bb_to_rect(x, y, w, h)
        shape = shape_predictor(frame_gray, rect)
        shape = shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        #x, y, w, h = rect_to_bb(rect)
        #cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        #for (x1, y1) in shape:
        #    cv.circle(frame, (x1, y1), 3, (0, 0, 255), -1)

        # filter nose region
        #angle = random.uniform(-bound*math.pi, bound*math.pi)
        #angle = bound * math.pi
        #frame = filter_nose(frame, shape, amount=amount, angle=angle, pinch=True)
        frame = filter_nose(frame, shape, amount=preset.amount, angle=preset.nose_angle, pinch=True)

        # filter mouth region
        #angle = random.uniform(-bound * math.pi, bound * math.pi)
        #angle = -bound * math.pi
        #frame = filter_smile(frame, shape, amount=amount, angle=angle, pinch=False)
        frame = filter_smile(frame, shape, amount=preset.amount, angle=preset.smile_angle, pinch=False)

        # add sunglasses over the eyes
        #p1 = ((shape[36][0] + shape[39][0]) // 2, (shape[37][1] + shape[38][1] + shape[40][1] + shape[41][1]) // 4)
        #p2 = ((shape[42][0] + shape[45][0]) // 2, (shape[43][1] + shape[44][1] + shape[46][1] + shape[47][1]) // 4)
        #frame = add_sunglasses(frame, p1, p2)

        # filter left eye region
        #angle = random.uniform(-bound * math.pi, bound * math.pi)
        #angle = -bound * math.pi
        #frame = filter_eye(frame, shape, left=True, amount=amount, angle=angle, pinch=False)
        frame = filter_eye(frame, shape, left=True, amount=preset.amount, angle=preset.left_angle, pinch=False)

        # filter right eye region
        #angle = random.uniform(-bound * math.pi, bound * math.pi)
        #angle = bound * math.pi
        #start = time.time()
        #frame = filter_eye(frame, shape, left=False, amount=amount, angle=-angle, pinch=False)
        frame = filter_eye(frame, shape, left=False, amount=preset.amount, angle=preset.right_angle, pinch=False)
        #print(time.time() - start)

    # show the output image with the face detections + facial landmarks
    #cv.imshow("Output", image)
    #cv.waitKey(0)

    '''
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

    '''
    print("detector points", points)
    if len(points) == 2:
        #frame = add_sunglasses(frame, points[0], points[1])
        frame = add_sunglasses(frame, np.add(points[0], (x, y)), np.add(points[1], (x, y)))
    '''

    #cv.imshow('Capture - Face detection', frame)
    #cv.imwrite('../examples/' + jpeg, frame)

    return frame


parser = argparse.ArgumentParser(description='Code for Face to Caricature application.')

# -- Haar cascades
parser.add_argument('--face_cascade', help='Path to face cascade.',
                    default='../data/haarcascades/haarcascade_frontalface_alt.xml')
#parser.add_argument('--eyes_cascade', help='Path to eyes cascade.',
#                    default='../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.',
                    default='../data/haarcascades/haarcascade_eye.xml')
parser.add_argument('--left_eye_cascade', help='Path to left eye cascade.',
                    default='../data/haarcascades/haarcascade_lefteye_2splits.xml')
parser.add_argument('--right_eye_cascade', help='Path to right eye cascade.',
                    default='../data/haarcascades/haarcascade_righteye_2splits.xml')
parser.add_argument('--smile_cascade', help='Path to smile cascade.',
                    default='../data/haarcascades/haarcascade_smile.xml')
parser.add_argument('--nose_cascade', help='Path to nose cascade.',
                    default='../data/haarcascades/haarcascade_mcs_nose.xml')

# -- facial landmarks
parser.add_argument('--lbfmodel', help='Path to LBFmodel.',
                    default='../data/facial_landmarks/lbfmodel.yaml')
parser.add_argument('--shape_predictor', help='Path to facial landmark predictor.',
                    default='../data/facial_landmarks/shape_predictor_68_face_landmarks.dat')
#parser.add_argument('--shape_predictor', help='Path to facial landmark predictor.',
#                    default='../data/facial_landmarks/shape_predictor_68_face_landmarks_GTX.dat')

# -- images and cameras
#parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
parser.add_argument('--image', help='Path to input image.', default='../data/images/face001.png')
#parser.add_argument('--image', help='Path to input image.', default='../data/images/face004.jpg')
parser.add_argument('--images', help='Path to input images.', default='../data/images/')
parser.add_argument('--overlay', help='Path to overlay image.', default='../data/overlays/sunglasses001.png')
#parser.add_argument('--overlay', help='Path to overlay image.', default='../data/overlays/eyepatch001.png')
parser.add_argument('--camera', help='Path to video.', default='../data/videos/visionface.avi')
parser.add_argument('--mat', help='path to widerface database (to be red)',
                    default='../data/wider_face_split/wider_face_train.mat')

# -- filter presets
parser.add_argument('--preset', help='Index number of the filter preset.', default=0)

args = parser.parse_args()

face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
left_eye_cascade_name = args.left_eye_cascade
right_eye_cascade_name = args.right_eye_cascade
smile_cascade_name = args.smile_cascade
nose_cascade_name = args.nose_cascade

#lbfmodel_name = args.lbfmodel

shape_predictor_name = args.shape_predictor

face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
left_eye_cascade = cv.CascadeClassifier()
right_eye_cascade = cv.CascadeClassifier()
smile_cascade = cv.CascadeClassifier()
nose_cascade = cv.CascadeClassifier()

#landmark_detector = cv.face.loadFacePoints()
#landmark_detector = cv.face_Facemark.fit()

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_name)

# -- 1. Load detectors and predictors
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
if not face_detector:
    print('--(!)Error loading face detector')
    exit(0)
if not shape_predictor:
    print('--(!)Error loading shape predictor')
    exit(0)

# -- 2. Read one image
image_name = args.image
#image = cv.imread(cv.samples.findFile(args.image), cv.IMREAD_GRAYSCALE)
image = cv.imread(cv.samples.findFile(image_name))
if image is None:
    print('Could not open or find the image:', args.image)
    exit(0)

# -- 3. Load image directory
image_path = args.images
files = []
for (dirpath, dirnames, filenames) in walk(image_path):
    files.extend(filenames)
    break
if len(files) == 0:
    print('Could not find files in path:', args.images)
    exit(0)

# -- 4. Read one overlay image
overlay_original = cv.imread(cv.samples.findFile(args.overlay), cv.IMREAD_UNCHANGED)
if overlay_original is None:
    print('Could not open or find the overlay image:', args.overlay)
    exit(0)

# -- 5. Load preset for filter settings
preset_index = args.preset
preset = FilterPreset(preset_index)
if preset is None:
    print('Could not load filter preset:', args.preset)
    exit(0)

'''
#-- 6. Load Widerface images from Matlab file
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
'''

'''
#-- 6. Read the video stream
camera_device = args.camera
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
'''

'''
while True:
    start = time.time()
    ret, frame = cap.read()
    output = detect_and_display(np.copy(frame), preset_index)
    cv.imshow('Caricature - press q to stop', output)

    if cv.waitKey(get_delay(start, fps=30)) & 0xFF == ord('q'):
        break  # q to quit
'''

for jpeg in files:
    stop = False
    start = time.time()
    print()
    print("Loading image:", jpeg)
    frame = cv.imread(image_path + jpeg)
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    '''
    h, w = frame.shape[:2]
    caricature = np.zeros((3*frame.shape[0], 3*frame.shape[1], frame.shape[2]), dtype=np.uint8)

    for i in range(9):
        #detect_and_display(frame)
        new_caricature = detect_and_display(np.copy(frame), index=i)

        if i < 3:
            caricature[:h, i*w:(i+1)*w] = new_caricature
        elif i < 6:
            caricature[h:2*h, (i-3)*w:(i-2)*w] = new_caricature
        else:
            caricature[2*h:3*h, (i-6)*w:(i-5)*w] = new_caricature

    output = cv.resize(caricature, (w, h))
    cv.imshow('Caricature - press q to stop', output)
    #cv.imwrite('../examples/collection.jpg', output)
    
    print(time.time() - start)
    '''

    '''
    output = detect_and_display(np.copy(frame), preset_index)
    #cv.imshow('Caricature - press q to stop', output)
    cv.imwrite('../examples/' + jpeg, output)
    
    # Wait for key press to update frame
    if cv.waitKey() & 0xFF == ord('q'):
        break  # q to quit
    '''

    for i in range(6):
        start = time.time()
        output = detect_and_display(np.copy(frame), i)

        cv.imshow('Caricature - press q to stop', output)
        #cv.imwrite('../examples/' + jpeg, output)

        print(time.time() - start)

        # Limit FPS
        #if cv.waitKey(get_delay(start, fps=30)) & 0xFF == ord('q'):
        if cv.waitKey(get_delay(start, fps=1)) & 0xFF == ord('q'):
            stop = True
            break  # q to quit

        # Wait for key press to update frame
        #if cv.waitKey() & 0xFF == ord('q'):
        #    stop = True
        #    break  # q to quit

    if stop:
        break

    '''
    while True:
        start = time.time()
        ret, frame = cap.read()
        output = detect_and_display(np.copy(frame), 3)
        cv.imshow('Caricature - press q to stop', output)

        if cv.waitKey(get_delay(start, fps=30)) & 0xFF == ord('q'):
        #if cv.waitKey(get_delay(start, fps=1)) & 0xFF == ord('q'):
            break  # q to quit

    break
    '''

    # Wait for key press to update frame
    #if cv.waitKey() & 0xFF == ord('q'):
    #    break  # q to quit

    #if cv.waitKey(10) == 27:
    #    break
