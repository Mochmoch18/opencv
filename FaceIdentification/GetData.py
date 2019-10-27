import cv2
from pip._vendor.distlib.compat import raw_input

facedetect=cv2.CascadeClassifier("/Users/user/PycharmProjects/Draw forms/"
                               "venv/lib/python3.7/site-packages/cv2/"
                               "data/haarcascade_frontalface_default.xml")
profiledetect=cv2.CascadeClassifier("/Users/user/PycharmProjects/Draw forms/"
                               "venv/lib/python3.7/site-packages/cv2/"
                               "data/haarcascade_profileface.xml")
cap=cv2.VideoCapture(0)
id = raw_input('Enter the user name : ')
sampleNum=0
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces=facedetect.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    profile=profiledetect.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    for x,y,w,h in faces:
        sampleNum=sampleNum+1
        print (sampleNum)
        cv2.imwrite("DataSet/"+id+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100)
    for x,y,w,h in profile:
        sampleNum=sampleNum+1
        print (sampleNum)
        cv2.imwrite("DataSet/"+id+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(1000)
    cv2.imshow('Face',frame)
    cv2.waitKey(1)
    if sampleNum > 20:
        break
cap.release()
cv2.destroyAllWindows()
