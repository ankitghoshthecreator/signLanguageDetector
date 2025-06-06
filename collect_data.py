import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Create data directory if not exists
os.makedirs("data", exist_ok=True)

# Ask user for the label
current_label = input("Enter label name: ").strip().upper()
samples_per_label = 100
sample_count = 0
data = []

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
print(f"Collecting data for: {current_label}")

while True:
    ret, frame = cap.read()
    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if len(landmarks) == 63:
                data.append(landmarks + [current_label])
                sample_count += 1
                print(f"Sample {sample_count} / {samples_per_label} for label '{current_label}'")

            mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.putText(image, f'Label: {current_label} | Count: {sample_count}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Collecting Hand Data', image)

    if sample_count >= samples_per_label:
        print("Data collection done.")
        break

    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Data collection interrupted.")
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV (append if exists)
columns = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
df_new = pd.DataFrame(data, columns=columns)

csv_path = "data/sign_data.csv"
if os.path.exists(csv_path):
    df_existing = pd.read_csv(csv_path)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(csv_path, index=False)
else:
    df_new.to_csv(csv_path, index=False)

print(f"Saved to {csv_path}")
