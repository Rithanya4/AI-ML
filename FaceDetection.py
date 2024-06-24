import cv2
alg="haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

cam=cv2.VideoCapture(0)

while True:
    _,img= cam.read()              ##Reading frame

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(gray, 1.3, 4)  ## 4 ->No.of neighbours

    for(x, y, w, h) in face:   #Traversing every frame
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,0,255),2)
    cv2.imshow("Face detection" , img)

    key=cv2.waitKey(10)
    if key == 27:     ##Esc key value is 27
        break
cam.release()
cv2.destroyAllWindows()
    
##For video input
##video_path = "name.mp4"
##cam = cv2.videoCapture(video_path)
