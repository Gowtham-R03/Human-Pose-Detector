import cv2
import mediapipe as mp #RGB
import time

############# Pose detection

mpPose = mp.solutions.pose
pose = mpPose.Pose()  #object
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("PoseEstimator/8.mp4")
frame_width = 1280
frame_height = 720
pTime = 0 #to find fps
while True:
    success, img = cap.read()
    img = cv2.resize(img, (frame_width, frame_height))
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)  #sending our video to object model(not marking only detecting)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)




    cTime = time.time() #current time
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)

    cv2.imshow("Video",img)
    cv2.waitKey(1)