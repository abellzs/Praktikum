import cv2
import dlib
import numpy as np

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) != 1:
        return None  # Return None when no face or more than one face is detected
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    top_lip_pts = landmarks[50:53].tolist() + landmarks[61:64].tolist()
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[1])

def bottom_lip(landmarks):
    bottom_lip_pts = landmarks[65:68].tolist() + landmarks[56:59].tolist()
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    if landmarks is None:  # Check if the result is None
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False

while True:
    ret, frame = cap.read()
    if not ret:
        break  # If no frame is captured, break the loop

    image_landmarks, lip_distance = mouth_open(frame)
    prev_yawn_status = yawn_status

    if lip_distance > 25:
        yawn_status = True
        cv2.putText(frame, "Subject is Yawning", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        output_text = " Yawn Count: " + str(yawns + 1)
        cv2.putText(frame, output_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
    else:
        yawn_status = False

    if prev_yawn_status == True and yawn_status == False:
        yawns += 1

    cv2.imshow('Live Landmarks', image_landmarks)
    cv2.imshow('Yawn Detection', frame)

    if cv2.waitKey(1) == 13:  # Exit on pressing 'Enter'
        break

cap.release()
cv2.destroyAllWindows()
