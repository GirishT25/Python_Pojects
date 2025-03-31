import cv2
import mediapipe as mp
import pyautogui

video = cv2.VideoCapture(0)
tip = [4, 8, 12, 16, 20]

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

accelerating = False
braking = False

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, imagevideo = video.read()
        
        if not ret:
            break

        imagevideo = cv2.flip(imagevideo, 1)
        image = cv2.cvtColor(imagevideo, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lmlist = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, c = image.shape
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmlist.append([id, cx, cy])
                
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(lmlist) == 21:
            fingers = []
            
            for id in range(1, 5):
                if lmlist[tip[id]][2] < lmlist[tip[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            if lmlist[tip[0]][1] > lmlist[tip[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            if fingers == [1, 1, 1, 1, 1]:
                if not accelerating:
                    pyautogui.keyDown("right")
                    accelerating = True
                braking = False
                pyautogui.keyUp("left")

            elif fingers == [0, 0, 0, 0, 0]:
                if not braking:
                    pyautogui.keyDown("left")
                    braking = True
                accelerating = False
                pyautogui.keyUp("right")

            else:
                pyautogui.keyUp("right")
                pyautogui.keyUp("left")
                accelerating = False
                braking = False

        cv2.imshow("Hand-Controlled Hill Climb Racing", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
            