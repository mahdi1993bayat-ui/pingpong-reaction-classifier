import os
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# ----------------------------------------
# 1. تنظیم اولیه
# ----------------------------------------
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = 'features_data'
label_map = {'G': 0, 'V': 1}
all_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.npy')]
random.shuffle(all_files)

# ----------------------------------------
# 2. تقسیم داده‌ها
# ----------------------------------------
train_split = int(0.6 * len(all_files))
val_split = int(0.8 * len(all_files))

train_files = all_files[:train_split]
val_files = all_files[train_split:val_split]
test_files = all_files[val_split:]

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []
test_filenames = []

for file in train_files:
    data = np.load(os.path.join(DATA_PATH, file))
    if data.shape[1] == 454:
        X_train.append(data)
        y_train.append(label_map[file[0]])

for file in val_files:
    data = np.load(os.path.join(DATA_PATH, file))
    if data.shape[1] == 454:
        X_val.append(data)
        y_val.append(label_map[file[0]])

for file in test_files:
    data = np.load(os.path.join(DATA_PATH, file))
    if data.shape[1] == 454:
        X_test.append(data)
        y_test.append(label_map[file[0]])
        test_filenames.append(file)

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

y_train = to_categorical(np.array(y_train))
y_val = to_categorical(np.array(y_val))
y_test = to_categorical(np.array(y_test))

# ----------------------------------------
# 3. ساخت مدل LSTM
# ----------------------------------------
cw = class_weight.compute_class_weight(class_weight='balanced',
                                       classes=np.unique(np.argmax(y_train, axis=1)),
                                       y=np.argmax(y_train, axis=1))
cw_dict = dict(enumerate(cw))

model = Sequential([
    LSTM(128, input_shape=(90, 454), return_sequences=False),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(2, activation='softmax')
])



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('lstm_best_model.h5', monitor='val_loss', save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=140,
    batch_size=32,
    callbacks=callbacks,
    class_weight=cw_dict,
    verbose=1
)

# ----------------------------------------
# 4. ارزیابی نهایی
# ----------------------------------------
preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, preds[:, 1])
cm = confusion_matrix(y_true, y_pred)

# ----------------------------------------
# 5. ذخیره PDF + PNG + CSV در دسکتاپ
# ----------------------------------------
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'KI')
os.makedirs(desktop_path, exist_ok=True)

pdf_path = os.path.join(desktop_path, 'lstm_eval_report.pdf')
png_path = os.path.join(desktop_path, 'lstm_confusion_matrix.png')
csv_path = os.path.join(desktop_path, 'lstm_prediction_results.csv')

# ذخیره فایل CSV نتایج پیش‌بینی
df = pd.DataFrame({
    'Filename': test_filenames,
    'True Label': y_true,
    'Predicted Label': y_pred,
    'Prob_G': preds[:, 0],
    'Prob_V': preds[:, 1]
})
df.to_csv(csv_path, index=False)

# تولید و ذخیره گزارش تصویری
with PdfPages(pdf_path) as pdf:
    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['G', 'V'], yticklabels=['G', 'V'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    pdf.savefig()
    plt.savefig(png_path, dpi=300)
    plt.close()

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    pdf.savefig()
    plt.close()

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    pdf.savefig()
    plt.close()

    # Summary Text
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    summary = (
        f"📊 LSTM Evaluation Summary (Test Set):\n\n"
        f"Accuracy:  {acc:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall:    {rec:.4f}\n"
        f"F1 Score:  {f1:.4f}\n"
        f"AUC-ROC:   {auc:.4f}\n\n"
        f"Confusion Matrix:\n{cm}"
    )
    ax.text(0.01, 0.95, summary, fontsize=12, va='top', family='monospace')
    pdf.savefig()
    plt.close()

# ----------------------------------------
# 6. چاپ مسیر فایل‌ها
# ----------------------------------------
print(f"\n✅ فایل‌ها ذخیره شدند در:\n📄 PDF: {pdf_path}\n🖼️ PNG: {png_path}\n📑 CSV: {csv_path}")
