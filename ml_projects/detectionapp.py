import easyocr as esr
import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np
class detector(cv2,imutils,np,esr):
    def __init__(self,img_path) -> None:
        self.img_path = img_path
    def finder(self,img_path):
        self.img_path=img_path
        self.img = cv2.imread(self.img_path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.bfilter = cv2.bilateralFilter(self.gray, 11, 17, 17) #Noise reduction
        self.edged = cv2.Canny(self.bfilter, 30, 200) #Edge detection
        self.keypoints = cv2.findContours(self.edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = imutils.grab_contours(self.keypoints)
        self.contours = sorted(self.contours, key=cv2.contourArea, reverse=True)[:10]
        self.location = None
        for self.contour in self.contours:
            self.approx = cv2.approxPolyDP(self.contour, 10, True)
            if len(self.approx) == 4:
                self.location = self.approx
                break
        return self.approx
    def reader(self,gray):
        self.gray=gray
        self.mask = np.zeros(self.gray.shape, np.uint8)
        (x,y) = np.where(self.mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        self.cropped_image = self.gray[x1:x2+1, y1:y2+1]
        reader = esr.Reader(['en'])
        self.result = reader.readtext(self.cropped_image)
        return self.result
         
    def drawer(self,img,text,approx):
        self.text=text
        img=self.img
        self.approx = approx
        self.text = self.text[0][-2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text=self.text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
        cv2.imshow('detector',cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

cap = cv2.VideoCapture('./data/car-vdo.mp4')
