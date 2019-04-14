# Face_Recognition
Face Recognition using LBPH and harcascade for Face detection
LBPH considers texture descriptor which is useful to symbolize Faces. Because face data can be split as compositions of patterns of micro textures. Basically LBPH is carried out in 3 stages they are
1. Feature extraction,
2. Matching,
3. Classification
The face recognition is carried out as stages first stage the image capturing and converting into grey scale then the haar features are checked if the features are their then it is considered as face
if not non face, after that the pixels are mapped and checked the face

 
  # RESULT
  The face detection and recognition is done using LBPH , the efficiency is up till 72% and the tilling of image is allowed till forty five degrees, the following images are the of face matched or recognized.

 


# CODE EXPLANATION

To setup opencv in python environment you will need these things ready ( match the versions to follow along with this tutorial),
<ul>
 <li>	Python 2.7 </li>
 <li>  Open cv 2.x </li>
<li>	Numpy library </li>
</ul>

First download python and install it in its default location (i.e c:/python27) after you have installed it download the opencv and extract it, go to “opencv/Build/python/2.7/x86” folder and copy “cv2.pyd” file to “c:/python27/Lib/site-packages/” folder.<br>
And now we are ready to use opencv in python. just one single problem is there, Opencv uses numpy library for its images so we have to install numpy library too <br> 
Go to start and type “cmd” you will see the command prompt icon right click on it and select “run as administrator”<br>
Now type<br>
“cd c:/python27/scripts/”
hit enter then type
“pip install numpy”


# DATA SET GENERATOR

Import the libraries

```
import numpy as np
import cv2

```
we can load the classifier now


```
detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
```
In the above line VideoCapture has one argument which is device id, for inbuilt webcam its usually ‘0’, and if we have some other webcam you can change that number so see that is your webcam’s Id

```ret, img = cap.read()```

In the above code we read the image from the video capture object using cap.read() method, it returns one status variable which is just True/False and the captured frame then we used imshow() method to display the image, here first argument is the window name and second argument is the image that we want to display, the third line we used waitKey(10) is used for a delay of 10 milisecond it is important for the imshow() method to work properly.

Before using the face detector we need comvert the captured image to Gray scale
```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
```

the above line will get the x,y and height,width of all the faces present in the captured image in a list, So now we have to loop through all the faces and draw rectangle there. the rectangle() the first argument is the input image in which we are going to draw the rectangles, second is the x,y coordinate of the face, then the height and weight, after that we are specifying the color of the line which is in the form of (blue,green,red) and you can adjust the value of each color, the range is 0-255, in this case its a green line, and the last argument is the line thickness

 we have marked the faces with green rectangles we can display them

```
  cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```
# TRAINER

import the opencv / cv2 library,
we will need the os to accress the file list in out dataset folder,
we also need to import the numpy library,
and we need to import the pillow / PIL library we installed before,
```

import cv2,os
import numpy as np
from PIL import Image
```
Now we need to initialize the recognizer and the face detector
```
recognizer = cv2.createLBPHFaceRecognizer()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
```
Load The Training Data
Ok, now we will are going to create a function which will grab the training images from the dataset folder, and will also get the corresponding Ids from its file name.So I am going to name this function “getImagesAndLabels”  we need the path of the dataset folder so we will provide the folder path as argument. So the function will be like this
```def getImagesAndLabels(path):```
So now inside this function we are going to do the following
⦁	Load the training images from dataset folder
⦁	capture the faces and Id from the training images
⦁	Put them In a List of Ids and FaceSamples  and return it
To load the image we need to create the paths of the image
    ```imagePaths=[os.path.join(path,f) for f in os.listdir(path)]```
this will get the path of each images in the folder.
now we need to create two lists for faces and Ids to store the faces and Ids
```
faceSamples=[]
Ids=[]	
```
Now we will loop the images using the image path and will load those images and Ids, we will add that in your lists
for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        ```pilImage=Image.open(imagePath).convert('L')```
        Now we are converting the PIL image into numpy array
        ```imageNp=np.array(pilImage,'uint8')```
        getting the Id from the image
        ```Id=int(os.path.split(imagePath)[-1].split(".")[1])```
        # extract the face from the training image sample
        ```faces=detector.detectMultiScale(imageNp)```
        #If a face is there then append that in the list as well as Id of it
        ```
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
                         ```
feed the data to the recognizer to train
```
faces,Ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')
```
Now if we run this code it will create a “trainner.yml” file inside the trainner folder,


# DETECTOR
Import important libraries
```
import cv2
import numpy as np
```
next we create a recognizer object using opencv library and load the training data (before that just sve your script in the same location where your “trainner” folder is located)
```
recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
```
Now we will create a cascade classifier using haar cascade for face detection, assuming u have the cascade file in the same location,
```
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
```
For video capture 
```
cam = cv2.VideoCapture(0)
```
Now we need a “font” that’s because we are going to write the name of that person in the image so we need a font for the text


```font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)```

start the main loop and do the following basic steps
⦁	Starts capturing frames from the camera object
⦁	Convert it to Gray Scale
⦁	Detect and extract faces from the images
⦁	Use the recognizer to recognize the Id of the user
⦁	Put predicted Id/Name and Rectangle on detected face

```
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
```
in the above two line the recognizer is predicting the user Id and confidence of the prediction respectively
in the next line we are writing the User ID in the screen below the face, which is (x, y+h) coordinate

we can add some more finishing touch like its showing user Id instead of the name,
and it cant handle unknown faces,
       
```if(conf<50):
            if(Id==1):
                Id="Paras"
            elif(Id==2):
                Id="Gorav"
        else:
            Id="Unknown"
    
    cv2.cv.PutText(cv2.cv.fromarray(im),str(Id), (x,y+h),font, 255)
    cv2.imshow('im',im) 
    if cv2.waitKey(10) &amp; 0xFF==ord('q'):
        break
   ```     
we need to close the camera and the windows. and we are done
``
cam.release()
cv2.destroyAllWindows()
```
