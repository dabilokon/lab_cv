# metrics_mobileNetV2.py

import torch
import matplotlib.pyplot as plt
import os
import time
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# 👇 СВОЇ ІМПОРТИ
from model_mobilenetv2 import MobileNetV2_Custom   # архітектура моделі
from data3 import val_loader                       # твій val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Використовується пристрій:", device)

# ---------- 1. Завантаження моделі MobileNetV2 ----------

model_path = "mobilenet_v2.pth"   # файл з вагами після тренування
model = MobileNetV2_Custom(num_classes=2, pretrained=False)  # pretrained=False, бо вже є свої ваги
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


# ---------- 2. Збір передбачень на валідації ----------

all_labels = []
all_preds = []
all_scores_blocked = []   # ймовірності для класу "blocked"

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        scores_blocked = probs[:, 0]     # клас "blocked" = індекс 0

        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_scores_blocked.extend(scores_blocked.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_scores_blocked = np.array(all_scores_blocked)


# ---------- 3. Основні метрики (1–4 + 5 = AUC) ----------

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, pos_label=0)
recall = recall_score(all_labels, all_preds, pos_label=0)
f1 = f1_score(all_labels, all_preds, pos_label=0)
cm = confusion_matrix(all_labels, all_preds)

# ROC-AUC (5-та метрика)
fpr, tpr, thresholds = roc_curve(all_labels, all_scores_blocked, pos_label=0)
roc_auc = auc(fpr, tpr)

print("\nМЕТРИКИ ДЛЯ МОДЕЛІ MobileNetV2 (клас 'blocked' як позитивний):")
print(f"1) Accuracy : {accuracy:.4f}")
print(f"2) Precision: {precision:.4f}")
print(f"3) Recall   : {recall:.4f}")
print(f"4) F1-score : {f1:.4f}")
print(f"5) ROC-AUC  : {roc_auc:.4f}")
print("Confusion Matrix:\n", cm)


# ---------- 4. Додаткові метрики (6–9) ----------

print("\n=== Додаткові метрики ефективності моделі ===")

# 6) Кількість параметрів
num_params = sum(p.numel() for p in model.parameters())
print(f"6) Кількість параметрів моделі: {num_params} ({num_params/1e6:.3f} млн)")

# 7) Розмір файла моделі
size_bytes = os.path.getsize(model_path)
size_mb = size_bytes / (1024 ** 2)
print(f"7) Розмір файла моделі '{model_path}': {size_mb:.2f} MB")

# 8–9) Час роботи
def measure_inference_time(model, loader, device):
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            _ = model(images)
    end = time.perf_counter()
    total = end - start
    return total, total / len(loader.dataset)

total_t, t_per_img = measure_inference_time(model, val_loader, device)
print(f"8) Загальний час передбачення на {len(val_loader.dataset)} зображень: {total_t:.4f} с")
print(f"9) Час на одне зображення: {t_per_img*1000:.4f} мс")


# ---------- 5. Барчарт додаткових метрик (Params, Size, Time) ----------

extra_names = ["Params (M)", "Size (MB)", "Time per Img (ms)", "Total Time (s)"]
extra_values = [
    num_params / 1e6,
    size_mb,
    t_per_img * 1000,
    total_t
]

plt.figure(figsize=(8, 5))
bars = plt.bar(extra_names, extra_values,
               color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
plt.title("Додаткові метрики ефективності моделі MobileNetV2")
plt.ylabel("Значення (у відповідних одиницях)")
plt.grid(axis="y", linestyle="--", alpha=0.5)

# ↓↓↓ Цифри всередині стовпчика (нижче вершини, щоб не залазили на назву)
for bar, value in zip(bars, extra_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value * 0.85,           # 85% висоти стовпчика
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.show()


# ---------- 6. ROC-крива ----------

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for class 'blocked' (MobileNetV2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ---------- 7. Confusion Matrix ----------

plt.figure(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["blocked", "free"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (MobileNetV2)")
plt.tight_layout()
plt.show()


# ---------- 8. Барчарт основних метрик (Accuracy, Precision, Recall, F1) ----------

metrics_names = ["Accuracy", "Precision (blocked)", "Recall (blocked)", "F1 (blocked)"]
metrics_values = [accuracy, precision, recall, f1]

plt.figure(figsize=(7, 5))
bars = plt.bar(metrics_names, metrics_values)
plt.ylim(0, 1.0)
plt.title("Основні метрики MobileNetV2")
plt.ylabel("Значення")

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h+0.02, f"{h:.2f}", ha="center")

plt.tight_layout()
plt.show()
