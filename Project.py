import cv2
import mediapipe as mp #RGB
import time
import PEstimaterModule as pm

cap = cv2.VideoCapture("PoseEstimator/8.mp4")
frame_width = 1280
frame_height = 720
pTime = 0  # to find fps
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = cv2.resize(img, (frame_width, frame_height))
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)


    cTime = time.time()  # current time
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow("Video", img)
    cv2.waitKey(1)