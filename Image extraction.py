#pip install opencv-python

import cv2

# path of file
data = cv2.VideoCapture("D:/360DigiTMG/Project/Data/IMG_3789.MOV")
currentframe = 0

while (True):
    success, image = data.read()
    if success:
        cv2.imwrite("D:/360DigiTMG/Project/Data/frame" + str(currentframe) + '.png', image)
        currentframe+=1
    else:
        break

data.release()       
cv2.destroyAllWindows()        