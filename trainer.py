import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
cascadePath=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
path='/home/hani/Detection visage/Reconizing/images'


def getImgID(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        #faces.append(imageNp)
        #Ids.append(Id)
        cv2.imshow("training",imageNp)
        cv2.waitKey(1000)
        # extract the face from the training image sample
        faces=cascadePath.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
           # faces.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return np.array(Ids),faces


faces,Ids = getImgID(path)
print(Ids,faces)
recognizer.train(faces, Ids)
recognizer.write('recognizer\training_data.yml')
cv2.destroyAllWindows()

