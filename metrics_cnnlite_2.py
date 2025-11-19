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


from model_cnnlite import CNNLite
from data3 import val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Використовується пристрій:", device)

# ---------- 1. Завантаження моделі ----------
model_path = "cnn_lite.pth"
model = CNNLite(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------- 2. Збір передбачень ----------
all_labels = []
all_preds = []
all_scores_blocked = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        scores_blocked = probs[:, 0]

        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_scores_blocked.extend(scores_blocked.cpu().numpy())

all_labels = torch.tensor(all_labels).numpy()
all_preds = torch.tensor(all_preds).numpy()
all_scores_blocked = torch.tensor(all_scores_blocked).numpy()

# ---------- 3. Базові метрики ----------
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, pos_label=0)
recall = recall_score(all_labels, all_preds, pos_label=0)
f1 = f1_score(all_labels, all_preds, pos_label=0)
cm = confusion_matrix(all_labels, all_preds)

print("\nМЕТРИКИ ДЛЯ МОДЕЛІ CNN-Lite (клас 'blocked' як позитивний):")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print("Confusion Matrix:\n", cm)

# ---------- 3.1. Додаткові метрики ----------
print("\n=== Додаткові метрики ефективності моделі ===")

# 1) Параметри
num_params = sum(p.numel() for p in model.parameters())
print(f"Кількість параметрів моделі: {num_params} ({num_params/1e6:.3f} млн)")

# 2) Розмір файлу
size_bytes = os.path.getsize(model_path)
size_mb = size_bytes / (1024 ** 2)
print(f"Розмір файла моделі '{model_path}': {size_mb:.2f} MB")

# 3–4) Час роботи
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
print(f"Загальний час передбачення на {len(val_loader.dataset)} зображень: {total_t:.4f} c")
print(f"Час на одне зображення: {t_per_img*1000:.4f} мс")

# ---------- ВІЗУАЛІЗАЦІЯ: 4 стовпчики додаткових метрик ----------
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
plt.title("Додаткові метрики ефективності моделі CNN-Lite")
plt.ylabel("Значення (у відповідних одиницях)")
plt.grid(axis="y", linestyle="--", alpha=0.5)

# ↓↓↓ ЗНИЖЕНІ ЦИФРИ (на 10% вище стовпчика)
    
for bar, value in zip(bars, extra_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value * 1.00,         # 90% висоти → текст знаходиться нижче
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10
    )
    

plt.tight_layout()
plt.show()

# ---------- 4. ROC-крива ----------
fpr, tpr, thresholds = roc_curve(all_labels, all_scores_blocked, pos_label=0)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for class 'blocked' (CNN-Lite)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ---------- 5. Confusion Matrix ----------
plt.figure(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["blocked", "free"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (CNN-Lite)")
plt.tight_layout()

# ---------- 6. Барчарт основних метрик ----------
metrics_names = ["Accuracy", "Precision (blocked)", "Recall (blocked)", "F1 (blocked)"]
metrics_values = [accuracy, precision, recall, f1]

plt.figure(figsize=(7, 5))
bars = plt.bar(metrics_names, metrics_values)
plt.ylim(0, 1.0)
plt.title("Основні метрики CNN-Lite")
plt.ylabel("Значення")
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h+0.02, f"{h:.2f}", ha="center")
plt.tight_layout()

plt.show()

