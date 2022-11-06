import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')
# test = eye_cascade.load('haarcascade/haarcascade_eye.xml')
# print(test)

# img = cv2.imread('crianca1.jpg')
img = cv2.imread('crianca2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detect_face = face_cascade.detectMultiScale(gray, 1.2, 1)

for(face_x, face_y, face_z, face_h) in detect_face:
    img2 = gray[face_y:face_y+face_h, face_x:face_x+face_z]
    detect_eye = eye_cascade.detectMultiScale(img2, 1.2, 1)
for (eye_x, eye_y, eye_z, eye_h) in detect_eye:
    eye1 = gray[face_y+eye_y:face_y+eye_y+eye_h, face_x+eye_x:face_x+eye_x+eye_z]
    ret, binary = cv2.threshold(eye1, 60, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('img', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
