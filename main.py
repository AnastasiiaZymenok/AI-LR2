import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Завантаження та підготовка даних
input_file = 'income_data.txt'
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

# Читання даних з файлу
try:
    with open(input_file, 'r') as f:
        for line in f.readlines():
except FileNotFoundError:
    print(f"Error: Could not find file '{input_file}'")
    exit(1)
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(0)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(1)
            count_class2 += 1

# Перетворення списку в масив numpy
X = np.array(X)
y = np.array(y)

# Кодування текстових даних у числові
label_encoders = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = label_encoder.fit_transform(X[:, i])
        label_encoders.append(label_encoder)

X = X_encoded.astype(float)

# Розділення даних на навчальні та тестові
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Функція для навчання та оцінки моделі
def train_and_evaluate(kernel_type):
    model = SVC(kernel=kernel_type)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    print(f"\n=== Kernel: {kernel_type} ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return accuracy, precision, recall, f1

# Навчання та оцінка для кожного ядра
results = {}
kernels = ['poly', 'rbf', 'sigmoid']

for kernel in kernels:
    results[kernel] = train_and_evaluate(kernel)

# Побудова графіку для порівняння
fig, ax = plt.subplots(figsize=(10, 6))

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors = ['blue', 'green', 'red', 'purple']

for i, metric_name in enumerate(metrics_names):
    values = [results[k][i] for k in kernels]
    ax.plot(kernels, values, label=metric_name, marker='o', color=colors[i])

ax.set_title('Порівняння продуктивності різних SVM ядер')
ax.set_xlabel('Тип ядра')
ax.set_ylabel('Значення метрики')
ax.legend()
plt.show()
