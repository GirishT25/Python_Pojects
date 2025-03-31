import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Initialize Webcam
video = cv2.VideoCapture(0)
video.set(3, 640)  # Set width
video.set(4, 480)  # Set height

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Audio Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Finger landmarks
tip = [4, 8, 12, 16, 20]

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    lmlist = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                lmlist.append([id, int(lm.x * w), int(lm.y * h)])

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if len(lmlist) == 21:
        fingers = [(lmlist[tip[i]][2] < lmlist[tip[i] - 2][2]) for i in range(1, 5)]
        fingers.append(lmlist[tip[0]][1] > lmlist[tip[0] - 1][1])

        # Control Volume
        current_vol = volume.GetMasterVolumeLevelScalar()
        if fingers == [1, 1, 1, 1, 1]:  # All fingers open
            volume.SetMasterVolumeLevelScalar(min(current_vol + 0.05, 1.0), None)
        elif fingers == [0, 0, 0, 0, 0]:  # All fingers closed
            volume.SetMasterVolumeLevelScalar(max(current_vol - 0.05, 0.0), None)
        elif fingers == [0, 1, 0, 0, 0]:  # Index finger up
            volume.SetMute(1, None)
        elif fingers == [1, 0, 0, 0, 0]:  # Thumb up
            volume.SetMute(0, None)

    cv2.imshow("Hand Gesture Volume Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
