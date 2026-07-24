import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# 1. بارگذاری مدل و داده
model = load_model('lstm_best_model.h5')
data = np.load('features_data/G_Ag_Xs_5_3.npy')
X = np.expand_dims(data, axis=0)

# 2. تعریف نام ویژگی‌ها
pose_names = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
pose_features = [f"{name}_{axis}" for name in pose_names for axis in ['x', 'y', 'z', 'v']]
face_features = [f"face_{i}_{axis}" for i in range(100) for axis in ['x', 'y', 'z']]
right_hand_features = [f"right_hand_{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]
left_hand_features = [f"left_hand_{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]
feature_names = pose_features + face_features + right_hand_features + left_hand_features

# 3. مفصل‌های دست
hand_landmarks = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

# 4. تابع برچسب‌گذاری ویژگی‌ها
def label_feature(f):
    if f.startswith("right_hand_"):
        idx, axis = f.replace("right_hand_", "").split("_")
        name = hand_landmarks[int(idx)] if int(idx) < len(hand_landmarks) else f"idx{idx}"
        return f"R_{name}_{axis}"
    elif f.startswith("left_hand_"):
        idx, axis = f.replace("left_hand_", "").split("_")
        name = hand_landmarks[int(idx)] if int(idx) < len(hand_landmarks) else f"idx{idx}"
        return f"L_{name}_{axis}"
    elif f.startswith("face_"):
        idx, axis = f.replace("face_", "").split("_")
        idx = int(idx)
        if 0 <= idx <= 9:
            area = "Center"
        elif 10 <= idx <= 20:
            area = "Eye_L"
        elif 21 <= idx <= 30:
            area = "Eye_R"
        elif 31 <= idx <= 40:
            area = "Brow_L"
        elif 41 <= idx <= 50:
            area = "Brow_R"
        elif 51 <= idx <= 70:
            area = "Nose"
        elif 71 <= idx <= 85:
            area = "Lip_Upper"
        elif 86 <= idx <= 100:
            area = "Lip_Lower"
        else:
            area = f"idx{idx}"
        return f"Face_{area}_{axis}"
    else:
        return f

# 5. محاسبه اهمیت ویژگی‌ها
def compute_importance(instance, model, epsilon=1e-2):
    original_pred = model.predict(np.expand_dims(instance, axis=0))[0]
    importances = np.zeros(instance.shape[1])
    for i in range(instance.shape[1]):
        modified = instance.copy()
        modified[:, i] += epsilon
        new_pred = model.predict(np.expand_dims(modified, axis=0))[0]
        diff = np.abs(new_pred - original_pred).sum()
        importances[i] = diff
    return importances

importances = compute_importance(X[0], model)

# 6. انتخاب 30 ویژگی مهم
sorted_idx = np.argsort(importances)[::-1][:30]
top_importances = importances[sorted_idx]
top_features = [feature_names[i] for i in sorted_idx]
top_labels = [label_feature(f) for f in top_features]

# 7. رنگ‌بندی
colors = []
for f in top_features:
    if f.startswith("face_"):
        colors.append("red")
    elif f.startswith("right_hand_"):
        colors.append("green")
    elif f.startswith("left_hand_"):
        colors.append("orange")
    elif any(f.startswith(p) for p in pose_names):
        colors.append("blue")
    else:
        colors.append("gray")

# 8. ذخیره روی دسکتاپ مک
output_path = os.path.expanduser("~/Desktop/KI/important_features_custom.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 9. رسم نمودار
plt.figure(figsize=(12, 8))
plt.barh(range(30), top_importances[::-1], color=colors[::-1])
plt.yticks(range(30), top_labels[::-1], fontsize=9)
plt.xlabel("Importance", fontsize=12)
plt.title("Top 30 Important Features with Detailed Face/Hand Labels", fontsize=14)
plt.tight_layout()
plt.savefig(output_path)
plt.close()

print(f"✅ تصویر نهایی با برچسب‌گذاری دقیق صورت و دست ذخیره شد در:\n{output_path}")
