import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from typing import Any
#Імпорт бібліотек 

iris: Any = load_iris()
X = iris.data  # Всі характеристики
y = iris.target  # Мітки класів
#Завантаження набору даних Iris та розділення його на характеристики та мітки класів

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(f"Дані успішно розділено на тренувальні та тестові набори.\nТренувальний набір: {len(X_train)} зразків\nТестовий набір: {len(X_test)} зразків")
#Розділення даних на тренувальний та тестовий набори

kernels = ['poly', 'rbf', 'sigmoid']
results = {}
# Оголошення ядер для перевірки

def train_and_evaluate(X_train, X_test, y_train, y_test, kernel_type):
    model = SVC(kernel=kernel_type, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
#Створення моделі SVM з вказаним ядром та навчання на тренувальному наборі даних
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = metrics.recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = metrics.f1_score(y_test, y_pred, average='macro', zero_division=1)
#Оцінка моделі на тестовому наборі даних за допомогою метрик 

    print(f"\n=== Ядро: {kernel_type} ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
#Виведення результатів оцінки моделі
    
    return accuracy, precision, recall, f1
#Повернення результатів оцінки моделі

for kernel in kernels:
    results[kernel] = train_and_evaluate(X_train, X_test, y_train, y_test, kernel)
# Перевірка моделей з різними ядрами та збереження результатів

fig, ax = plt.subplots(figsize=(10, 6))
#Створення графіку для візуалізації результатів

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors = ['blue', 'green', 'red', 'purple']

for i, metric_name in enumerate(metrics_names):
    values = [results[kernel][i] for kernel in kernels]
    ax.plot(kernels, values, label=metric_name, marker='o', color=colors[i])
#Накладання метрик на графік

ax.set_title('Порівняння продуктивності різних SVM ядер на наборі даних Iris')
ax.set_xlabel('Тип ядра')
ax.set_ylabel('Значення метрики')
ax.legend()
plt.show()
#Налаштування та відображення графіку