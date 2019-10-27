import cv2
import glob
import numpy as np
import os
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create()
def getImageWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    if path + '/.DS_Store' in imagePaths:
        imagePaths.remove(path + '/.DS_Store')
    print(imagePaths)
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[0])
        print(ID)
        faces.append(faceNp)
        IDs.append(ID)
        #cv2.imshow("Training",faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces
Ids,faces=getImageWithID('DataSet')
recognizer.train(faces,Ids )
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()

