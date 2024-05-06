import cv2
import dlib
import numpy as np

# File path to the shape predictor model
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
# Scaling factor for resizing images
SCALE_FACTOR = 1
# Amount to feather edges for mask
FEATHER_AMOUNT = 11

# Define points for different parts of the face
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used for aligning the images
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
# Points to overlay for the mask
OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS, NOSE_POINTS + MOUTH_POINTS]

# Amount of blurring for color correction
COLOUR_CORRECT_BLUR_FRAC = 0.6

# Setup dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im, dlibOn=True):
    if dlibOn:
        rects = detector(im, 1)
        if len(rects) != 1:
            return None
        return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    else:
        # Fallback for Haar Cascade (if needed, load your cascade here)
        rects = cascade.detectMultiScale(im, 1.3, 5)
        if len(rects) != 1:
            return None
        x, y, w, h = rects[0]
        rect = dlib.rectangle(x, y, x + w, y + h)
        return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def read_im_and_landmarks(fname, dlibOn=True):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    if im is None:
        print("Image not found:", fname)
        return None, None
    im = cv2.resize(im, (int(im.shape[1] * SCALE_FACTOR), int(im.shape[0] * SCALE_FACTOR)))
    landmarks = get_landmarks(im, dlibOn)
    return im, landmarks

def face_swap(img, name, dlibOn=True):
    s = get_landmarks(img, dlibOn)
    if s is None:
        print("No or too many faces")
        return img
    im2, landmarks2 = read_im_and_landmarks(name, dlibOn)
    if im2 is None or landmarks2 is None:
        print("Failed to read landmarks for image:", name)
        return img
    # Additional face swapping logic here...
    # This is a placeholder, you should add your warping, masking, and color correction here

    return img  # This should be replaced with the swapped face image

cap = cv2.VideoCapture(0)
filter_image = "faceandeye3.jpg"  # Change to your actual filter image path

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    frame = cv2.resize(frame, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
    frame = cv2.flip(frame, 1)
    swapped_frame = face_swap(frame, filter_image)
    cv2.imshow('Our Amazing Face Swapper', swapped_frame)
    if cv2.waitKey(1) == 13:  # Press Enter to exit
        break

cap.release()
cv2.destroyAllWindows()
