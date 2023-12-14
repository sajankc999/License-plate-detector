import easyocr as esr
import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np
class detector():
    
    def finder(img):
        img=img
        #img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
        edged = cv2.Canny(bfilter, 30, 200) #Edge detection
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        return approx
    def reader(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, np.uint8)
        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        reader = esr.Reader(['en'])
        result = reader.readtext(cropped_image)
        return result
         
    def drawer(self,img,text,approx):
        self.text=text
        self.img=img
        self.approx = approx
        self.text = self.text[0][-2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text=self.text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
        cv2.imshow('detector',cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

cap = cv2.VideoCapture('D:\ml_projects\license_detection\data\car-vdo.mp4') 
  
# Check if camera opened successfully 
if (cap.isOpened()== False): 
    print("Error opening video file") 
  
# Read until video is completed 
while(cap.isOpened()): 
      
# Capture frame-by-frame 
    ret, frame = cap.read() 
    if ret == True: 
    # finding the plate 
        
        location=detector.finder(img=frame)
        text = detector.reader(img=frame)
        print(location)
          
    # Press Q on keyboard to exit 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
  
# Break the loop 
    else: 
        break
  
# When everything done, release 
# the video capture object 
cap.release() 
  
# Closes all the frames 
cv2.destroyAllWindows() 