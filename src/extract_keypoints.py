import os
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
import random

# مسیرها
RAW_PATH = 'raw_videos'
DATA_PATH = 'AUGMENTED_KEYPOINTS'  # 👈 تغییر نام پوشه ذخیره‌سازی
MAX_FRAMES = 90
label_map = {'G': 0, 'V': 1}

# ساخت پوشه ذخیره‌سازی
os.makedirs(DATA_PATH, exist_ok=True)

# MediaPipe setup
mp_holistic = mp.solutions.holistic

# ⬇️ تابع Augmentation فریم‌ها
def augment_frame(frame):
    if random.random() < 0.5:
        frame = cv2.flip(frame, 1)

    if random.random() < 0.5:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value = random.uniform(0.5, 1.5)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value, 0, 255)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if random.random() < 0.3:
        angle = random.randint(-15, 15)
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        frame = cv2.warpAffine(frame, M, (w, h))

    # ✅ افزودن نویز Gaussian (Geräusche)
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

    return frame

# ⬇️ استخراج keypoints
def extract_keypoints(results):
    important_pose_idxs = [0, 11, 12, 13, 14, 15, 16]
    pose = (
        np.array([[results.pose_landmarks.landmark[i].x,
                   results.pose_landmarks.landmark[i].y,
                   results.pose_landmarks.landmark[i].z,
                   results.pose_landmarks.landmark[i].visibility]
                  for i in important_pose_idxs])
        if results.pose_landmarks else np.zeros((len(important_pose_idxs), 4))
    )

    face = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark[:100]])
        if results.face_landmarks else np.zeros((100, 3))
    )

    rh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
        if results.right_hand_landmarks else np.zeros((21, 3))
    )

    lh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
        if results.left_hand_landmarks else np.zeros((21, 3))
    )

    return np.concatenate([pose.flatten(), face.flatten(), rh.flatten(), lh.flatten()])

# پردازش ویدیوها
with mp_holistic.Holistic(static_image_mode=False) as holistic:
    for filename in tqdm(os.listdir(RAW_PATH), desc="🎥 Extracting keypoints with augmentation"):
        if not filename.endswith(('.mp4', '.mov')):
            continue
        label = label_map.get(filename[0])
        if label is None:
            continue

        cap = cv2.VideoCapture(os.path.join(RAW_PATH, filename))
        sequence = []

        while cap.isOpened() and len(sequence) < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            frame = augment_frame(frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
        cap.release()

        # padding در صورت کمبود فریم
        while len(sequence) < MAX_FRAMES:
            sequence.append(sequence[-1])

        # ⬇️ ذخیره در پوشه جدید
        np.save(
            os.path.join(DATA_PATH, filename.replace(".mp4", ".npy").replace(".mov", ".npy")),
            np.array(sequence)
        )
