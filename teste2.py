from ast import List
import cv2
import numpy as np
from dataclasses import dataclass
import statistics
# Montar Classes


@dataclass
class FaceRoi:
    x: int
    y: int
    w: int
    h: int


@dataclass
class EyeRoi:
    x: int
    y: int
    w: int
    h: int


class Recognize:

    @classmethod
    def _recognize_face_roi(self, image):
        # Returns a tuple in cases that cannot define faces ROI
        face = face_cascade.detectMultiScale(image, 1.1, 10)
        if type(face) is tuple:
            print("Erro ao definir face")
            return False

            # Drawing rectangle around the face
        for(x, y,  w,  h) in face:
            face = FaceRoi(
                x=x,
                y=y,
                w=w,
                h=h
            )
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            print(f'\n\nX: {x}\nY: {y}\nW: {w}\nH: {h}')

        print("Sucesso ao definir Face")
        return face

    @classmethod
    def _recognize_eye_roi(self, image, face: FaceRoi):
        roi_face_gray = image[face.y:(face.y+face.h), face.x:(face.x+face.w)]
        roi_face_color = img[face.y:(face.y+face.h), face.x:(face.x+face.w)]
        eyes_roi = eye_cascade.detectMultiScale(roi_face_gray, 1.1, 10)
        eyes_list = []
        if type(eyes_roi) is tuple:
            print("Erro ao definir olhos")
            return False

        for (x_eye, y_eye, w_eye, h_eye) in eyes_roi:
            if x_eye == 74:
                continue
            eye = EyeRoi(
                x=x_eye,
                y=y_eye,
                w=w_eye,
                h=h_eye
            )
            eyes_list.append(eye)

            print(f'\n\nx_eye: {x_eye}\ny_eye: {y_eye}\nw_eye: {w_eye}\nh_eye: {h_eye}')
        
        
        # TRATA QUANDO TEM MAIS DE 3 ROIS
        if len(eyes_list) >= 3:
            print(f'\n\nEYES LIST: {len(eyes_list)}')
            # remove roi que estejam discrepantes
            roi_edge_list = []
            for eye in eyes_list:
                edge = eye.w
                roi_edge_list.append(edge)
            
            to_remove_list = []

            for edge in roi_edge_list:
                variacao_edge = (edge-statistics.mean(roi_edge_list))/edge*100
                print(f"\nVARIATION: {type(variacao_edge)}")
                print(f"\nEDGE VARIATION: {abs(variacao_edge)}")
                if int(abs(variacao_edge)) > 40:
                    print('\nREMOVER')
                    to_remove_list.append(edge)
                    roi_edge_list.remove(edge)
            
            
            for eye in eyes_list:
                if eye.w in to_remove_list:
                    eyes_list.remove(eye)

        # print(f'\nMIN VALUE: {min(roi_edge_list)} \nMAX VALUE: {max(roi_edge_list)} \nMEAN: {statistics.mean(roi_edge_list)}')
        # print(f'\nVAR: {np.var(roi_edge_list)}')

        # print(f"\nFINA LIST: {roi_edge_list}")

        # for (x_eye, y_eye, w_eye, h_eye) in eyes_list:
        #     cv2.rectangle(
        #         roi_face_color,
        #         (x_eye, y_eye),
        #         (x_eye+w_eye, y_eye+h_eye),
        #         (0, 255, 0),
        #         2
        #     )
        #     roi_eye_gray = roi_face_gray[y_eye:y_eye+h_eye, x_eye:x_eye+w_eye]
        #     roi_eye_color = roi_face_color[y_eye:y_eye+h_eye, x_eye:x_eye+w_eye]
        #     roi_bgr = cv2.cvtColor(roi_eye_gray, cv2.COLOR_GRAY2BGR)
        #     _, thresh = cv2.threshold(roi_eye_gray, 92, 255, cv2.THRESH_BINARY_INV)
        #     morph_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        #     morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel)


        #     # plt.hist(morph_open.ravel(),256,[0,256]); plt.show()
        #     n_white_pix = np.sum(morph_open == 255)
        #     print(f"n_white_pix {n_white_pix}")
        #     cv2.imshow("roi_eye_gray", morph_open)
        #     # cv2.imshow("morph_open", morph_open)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        # FAZER APENAS COM 2 ROIS
        white_pixels_list = []
        for eye in eyes_list:
            cv2.rectangle(
                roi_face_color,
                (eye.x, eye.y),
                (eye.x+eye.w, eye.y+eye.h),
                (0, 255, 0),
                2
            )
            roi_eye_gray = roi_face_gray[eye.y:eye.y+eye.h, eye.x:eye.x+eye.w]
            roi_eye_color = roi_face_color[eye.y:eye.y+eye.h, eye.x:eye.x+eye.w]
            roi_bgr = cv2.cvtColor(roi_eye_gray, cv2.COLOR_GRAY2BGR)
            _, thresh = cv2.threshold(roi_eye_gray, 92, 255, cv2.THRESH_BINARY_INV)
            morph_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel)


            # DAR UM JEITO DE CALCULAR A VARIACAO
            n_white_pix = np.sum(morph_open == 255)
            white_pixels_list.append(n_white_pix)
            print(f"n_white_pix {n_white_pix}")
            cv2.imshow("roi_eye_gray", morph_open)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print("Sucesso ao definir olhos")
        return face, white_pixels_list


img = cv2.imread("crianca2.png")
# img = cv2.imread("mardita2.jpg")
# img = cv2.imread("mardita.jpg")
# variacao dos pixeis brancos
variacao_mardita2 = (4692-2628)/4692*100
variacao_crianca2 = (462-170)/462*100
variacao_mardita = (5334-4712)/5334*100
print(f"\n VARIACAO {variacao_mardita2}\n")
print(f"\n VARIACAO {variacao_crianca2}\n")
print(f"\n VARIACAO {variacao_mardita}\n")

face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')
kernel = np.ones((3, 3), np.uint8)
# Creating an object faces
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

recognizer = Recognize()
recognized_face_roi = recognizer._recognize_face_roi(gray_img)
recognized_face_roi, white_pixels_list = recognizer._recognize_eye_roi(
    image=gray_img, 
    face=recognized_face_roi
)

max_value = max(white_pixels_list)
min_value = min(white_pixels_list)
percent_variation = (max_value-min_value)/max_value*100
print(f"\n percent_variation {percent_variation}\n")

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
