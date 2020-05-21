from keras.models import load_model
import PIL
from PIL import Image
import cv2
import numpy as np
import os

#initialize model
model = load_model(os.getcwd()+"/gesturesModelAug.h5")
cam = cv2.VideoCapture(os.getcwd()+"/test_vid.mov")
subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(history=20,backgroundRatio=0.1)
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2

Lookup = {
    0:'Palm',
    1:'L',
    2:'Fist',
    3:'Fist Turned',
    4:'Thumb Pointed Out',
    5:'Index Finger Pointed Out',
    6:'Okay',
    7:'Palm Turned',
    8:'C',
    9:'down'
}
while True:
    _, frame = cam.read()

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Skin Color
    lower = np.array([6,100,0])
    upper = np.array([16,255,255])
    mask = cv2.inRange(hsv_frame,lower,upper)
    pFrame = cv2.bitwise_and(frame, frame, mask = mask)
    cv2.imshow("Masked",pFrame)
    pFrame = Image.fromarray(pFrame)
    pFrame = pFrame.resize((320,120)).convert('L')
    pFrame.save('local_test.jpg')
    pFrame = np.array(pFrame,dtype='float32')
    pFrame = pFrame.reshape((120,320))
    pFrame /= 225
    pFrame = pFrame.reshape((1,120,320,1))

    prediction = model.predict(pFrame).tolist()[0]

    maxPred = max(prediction)
    print(maxPred)
    if maxPred > 0.9:
        frame = cv2.putText(frame,Lookup[prediction.index(maxPred)]+" - Confidence: "+str(maxPred), org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 

    cv2.imshow("",frame)

    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

