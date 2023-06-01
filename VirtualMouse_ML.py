import cv2
import numpy
import HandTrackingModule_ML as htm
import autopy

#width and height parameters
wDefault = 640
hDefault = 480
wScreen, hScreen = autopy.screen.size()
frame = 120

#video capture parameters
capture = cv2.VideoCapture(0)
capture.set(3, wDefault) #width property id is 3
capture.set(4, hDefault) #height property id is 4

detector = htm.handDetector(maxHands=1)

smooth = 4
pX = 0
pY = 0
cX = 0
cY = 0

while True:
    #1. Find hand landmarks
    success, img = capture.read()
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img)

    #2. Get index & middle fingers
    if len(landmarkList) != 0:
        #index finger coordinates (x/y1)
        x1, y1 = landmarkList[8][1:]
        #middle finger coordinates (x/y2)
        x2, y2 = landmarkList[12][1:]

        #3. Check which fingers are up
        fingerTips = detector.fingersUp()
        cv2.rectangle(img, (frame, frame), (wDefault - frame, hDefault - frame), (255, 0, 255), 2)

        #4. Index finger only = movement mode
        if fingerTips[1] == 1 and fingerTips[2] == 0:
            #4a. Convert coordinates from default resolution to actual resolution
            x3 = numpy.interp(x1, (frame, wDefault - frame), (0, wScreen)) #convert x1 from the first range to the second
            y3 = numpy.interp(y1, (frame, hDefault - frame), (0, hScreen))
            #4b. Smooth values
            cX = pX + (x3 - pX)/smooth
            cY = pY + (y3 - pY)/smooth
            #4c. Move mouse
            autopy.mouse.move(wScreen - cX, cY)
            cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
            pX = cX
            pY = cY

    #5. Index + Middle fingers = clicking mode
        #5a. Find distance b/w fingers
        if fingerTips[1] == 1 and fingerTips[2] == 1:
            length, img, linePoints = detector.findDistance(8, 12, img)
            # 5b. Click mouse if distance is short
            print(linePoints)
            if(length < 32):
                cv2.circle(img, (linePoints[4], linePoints[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    #6. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)