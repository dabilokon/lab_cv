# Запускаємо навчання (кілька епох)
import torch
import torch.nn as nn
from model_cnnlite import CNNLite
from train_utils import train_one_epoch, eval_one_epoch

# Завантаження даталоадерів
from data3 import train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Використовується пристрій:", device)

model = CNNLite(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

    print(f"Епоха {epoch+1}/{num_epochs}: "
          f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
          f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

# збережемо модель
torch.save(model.state_dict(), "cnn_lite.pth")
print("Модель CNN-Lite збережено в cnn_lite.pth")
