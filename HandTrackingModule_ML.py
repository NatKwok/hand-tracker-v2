import cv2
import mediapipe
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.complexity = complexity

        # initializes a hand object - a component of the handDetector
        self.handSolutions = mediapipe.solutions.hands
        self.hands = self.handSolutions.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackingCon)
        # draws hand landmark points
        self.drawingUtils = mediapipe.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converts image to RGB so that Hands can use it
        self.results = self.hands.process(imgRGB) # built-in function in Hands ("process") that specifies the image
        # check if there are multiple hands, and extract 1 by 1, but only if there is something in the result
        if self.results.multi_hand_landmarks:  # hand landmarks is built-in representation of all hand landmarks
            for handLms in self.results.multi_hand_landmarks:
                # draws on the original BGR image, for one hand, HAND_CONNECTIONS includes lines between points
                if draw:
                    self.drawingUtils.draw_landmarks(img, handLms, self.handSolutions.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        #list of all landmarks
        xList = []
        yList = []
        box = []
        self.landmarkList = []
        #if there are landmarks
        if self.results.multi_hand_landmarks:
            #specify the landmark
            myHand = self.results.multi_hand_landmarks[handNo]
            # get the data from the hand - get landmarks using the built-in .landmark designation
            # each landmark has an x, y, and z coordinate, interpreted as a ratio of the image
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                # to get the x & y coordinates in pixels (only x & y matter for the purposes of the mouse), multiply by image width & height
                h, w, c = img.shape
                pixelX = int(lm.x * w)
                pixelY = int(lm.y * h)
                #enter landmarks into the list
                xList.append(pixelX)
                yList.append(pixelY)
                self.landmarkList.append([id, pixelX, pixelY])
                # draw circle for landmark 8, the tip of the index finger
                if id == 8 & draw == True:
                    cv2.circle(img, (pixelX, pixelY), 15, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            box = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0,255,0), 2)

        return self.landmarkList, box

    def fingersUp(self):
        fingerTips = []
        #Thumb
        if self.landmarkList[self.tipIds[0]][1] > self.landmarkList[self.tipIds[0] - 1][1]:
            fingerTips.append(1)
        else:
            fingerTips.append(0)
        #Fingers
        for id in range(1,5):
            if self.landmarkList[self.tipIds[id]][2] < self.landmarkList[self.tipIds[id] - 2][2]:
                fingerTips.append(1)
            else:
                fingerTips.append(0)

        return fingerTips

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.landmarkList[p1][1:]
        x2, y2 = self.landmarkList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255,0,255), t)
            cv2.circle(img, (x1, y1), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]



def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) > 0:
            print(lmList[7])

        cv2.imshow("Image", img)
        cv2.waitKey(200)

if __name__ == "__main__":
    main()