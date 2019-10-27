import cv2
facedetect=cv2.CascadeClassifier("/Users/user/PycharmProjects/Draw forms/"
                               "venv/lib/python3.7/site-packages/cv2/"
                               "data/haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainingData.yml")
id=0

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces=facedetect.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if (id==1):
            id="MyAhmed"
        elif (id==2):
            id="Yousra"
        elif (id==3):
            id="Younes"
        elif (id==4):
            id="Loubna"
        cv2.putText(frame,str(id),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        cv2.waitKey(10)
    cv2.imshow('Face',frame)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
