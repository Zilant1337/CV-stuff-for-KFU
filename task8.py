import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Загрузка данных
digits = load_digits()
X = digits.data  # Данные (1797, 64)
y = digits.target  # Метки (1797,)

# Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение данных на обучающую, валидационную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Преобразование данных в тензоры PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Создание DataLoader
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Определение модели
class SimpleVGG(nn.Module):
    def __init__(self):
        super(SimpleVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 1x8x8 -> 8x8x8
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),  # 8x8x8 -> 8x8x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8x8 -> 8x4x4
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 8x4x4 -> 16x4x4
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # 16x4x4 -> 16x4x4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x4x4 -> 16x2x2
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 2 * 2, 100),  # 16x2x2 -> 100
            nn.ReLU(),
            nn.Linear(100, 10),  # 100 -> 10
            # nn.Softmax(1)
        )

    def forward(self, x):
        x = x.view(-1, 1, 8, 8)  # Преобразуем входные данные в 4D тензор (batch_size, channels, height, width)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Выравниваем тензор для полносвязного слоя
        x = self.classifier(x)
        return x


# Инициализация модели, функции потерь и оптимизатора
model = SimpleVGG()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Обнуляем градиенты
        optimizer.zero_grad()

        # Прямой проход
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Валидация
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {running_loss / len(train_loader):.4f}, "
          f"Val Loss: {val_loss / len(val_loader):.4f}, "
          f"Val Accuracy: {100 * correct / total:.2f}%")

# Оценка на тестовой выборке
model.eval()
test_correct = 0
test_total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Test Accuracy: {100 * test_correct / test_total:.2f}%")

# Матрица ошибок
conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(conf_matrix)