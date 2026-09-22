# 🧠 PyTorch Neural Networks: Classification & Regression

Навчальний репозиторій з двома проєктами на **PyTorch**, які демонструють побудову, навчання та оцінку нейронних мереж для задач класифікації та регресії.

## 🎯 Мета проєкту

Репозиторій створений для практичного вивчення основ глибокого навчання за допомогою PyTorch:
- побудова архітектури нейронної мережі (`nn.Module`, `nn.Sequential`);
- підготовка даних (train/test split, масштабування ознак);
- налаштування функції втрат та оптимізатора;
- цикл навчання (forward pass, backward pass, оновлення ваг);
- оцінка якості моделі на тестових даних.

## 🛠️ Технології

| Технологія | Призначення |
|---|---|
| **Python** | мова програмування |
| **PyTorch** (`torch`, `torch.nn`) | побудова та навчання нейронних мереж |
| **Scikit-learn** | генерація/завантаження датасетів, `train_test_split`, `StandardScaler` |
| **Jupyter / Google Colab** | середовище розробки |

---

## 📌 Завдання 1: Класифікація `make_moons` (бінарна класифікація)

Синтетичний датасет із двох переплетених "півмісяців" — класична нелінійно роздільна задача, яка вимагає прихованих шарів для розв'язання.

### Підготовка даних

```python
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
```

### Архітектура моделі

```python
class MPCClassifier_neural(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 200),    # 2 вхідні ознаки (x1, x2)
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = MPCClassifier_neural()
```

### Функція втрат та оптимізатор

```python
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### Цикл навчання

```python
epoches = 300

for epoch in range(epoches):
    optimizer.zero_grad()
    y_predict = model(X_train_tensor)
    loss = criterion(y_predict, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Епохи {epoch + 1} помилка {loss}')
```

### Результат

Помилка (BCE Loss) стабільно спадала протягом навчання:

| Епохи | Помилка |
|---|---|
| 10 | 0.6582 |
| 100 | 0.4875 |
| 200 | 0.3763 |
| 300 | 0.3285 |

---

## 📌 Завдання 2: Регресія `California Housing`

Датасет на основі перепису населення Каліфорнії 1990 року. Мета — передбачити середню ціну будинку в окрузі на основі 8 числових ознак (дохід населення, вік будинків, кількість кімнат тощо).

### Підготовка даних

```python
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
```

### Архітектура моделі

```python
class MLPRegression_neural(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        return self.model(x)

model = MLPRegression_neural()
```

### Функція втрат та оптимізатор

```python
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### Цикл навчання

```python
epoches = 1000

for epoch in range(epoches):
    optimizer.zero_grad()
    y_predict = model(X_train_tensor)
    loss = criterion(y_predict, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Епохи {epoch + 1} та помилка {loss}')
```

### Оцінка на тестових даних

```python
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    test_mse = criterion(y_test_pred, y_test_tensor)
    test_mae = nn.L1Loss()(y_test_pred, y_test_tensor)

print(f"Тестовий MSE: {test_mse.item():.4f}")
print(f"Тестовий MAE: {test_mae.item():.4f}")
```

### Результат

| Метрика | Значення |
|---|---|
| Тестовий MSE | 0.4442 |
| Тестовий MAE | 0.4758 (у сотнях тисяч доларів) |

---

## 📂 Структура репозиторію

```
├── task1_moons_classification.ipynb   # Бінарна класифікація (make_moons)
├── task2_california_housing.ipynb     # Регресія (California Housing)
└── README.md
```

## 🚀 Запуск

```bash
pip install torch scikit-learn
```

Відкрий ноутбуки у Jupyter Notebook або Google Colab та виконай комірки послідовно.

## 📈 Можливі покращення

- Використати `Adam` замість `SGD` для швидшої збіжності;
- Додати `Dropout` / `BatchNorm` для регуляризації;
- Підбір гіперпараметрів (learning rate, розмір прихованих шарів);
- Візуалізація межі рішення для `make_moons` та графіків втрат.

---

*Проєкт створено в рамках вивчення глибокого навчання на PyTorch.*
