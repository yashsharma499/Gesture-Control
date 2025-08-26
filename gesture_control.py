import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Disable PyAutoGUI failsafe (otherwise mouse may freeze if goes to corner)
pyautogui.FAILSAFE = False

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Screen resolution
screen_w, screen_h = pyautogui.size()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for natural movement
    frame = cv2.flip(frame, 1)

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmarks = hand_landmarks.landmark

            # Index finger tip
            index_x = int(landmarks[8].x * screen_w)
            index_y = int(landmarks[8].y * screen_h)

            # Move mouse
            pyautogui.moveTo(index_x, index_y)

            # Thumb tip
            thumb_x = int(landmarks[4].x * screen_w)
            thumb_y = int(landmarks[4].y * screen_h)

            # Middle finger tip
            middle_x = int(landmarks[12].x * screen_w)
            middle_y = int(landmarks[12].y * screen_h)

            # Euclidean distances
            dist_index_thumb = np.hypot(index_x - thumb_x, index_y - thumb_y)
            dist_middle_thumb = np.hypot(middle_x - thumb_x, middle_y - thumb_y)

            # Left click: pinch index + thumb
            if dist_index_thumb < 40:
                pyautogui.click()
                cv2.putText(frame, "Left Click", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Right click: pinch middle + thumb
            elif dist_middle_thumb < 40:
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show webcam feed
    cv2.imshow("Gesture Control", frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
