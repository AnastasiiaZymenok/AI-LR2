import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt
#Імпорт необхідних бібліотек

def load_data(file_path):
    try:
        X = []
        y = []
        count_class1 = 0
        count_class2 = 0
        max_datapoints = 25000
#зчитування даних з файлу та створення списків для даних та міток, лічильники для класів та максимальна кількість даних
        with open(file_path, 'r') as f:
            lines = f.readlines()
            print(f"Файл завантажений успішно. Кількість рядків: {len(lines)}")

            for line in lines:
                if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                    break
                if '?' in line:
                    continue

                data = line.strip().split(', ')
#Зчитування рядків з файлу, пропуск рядків з пропущеними значеннями та розділення рядків на дані та мітки

                if data[-1] == '<=50K' and count_class1 < max_datapoints:
                    X.append(data[:-1])
                    y.append(0)
                    count_class1 += 1
                elif data[-1] == '>50K' and count_class2 < max_datapoints:
                    X.append(data[:-1])
                    y.append(1)
                    count_class2 += 1
#створює мітку класу та додає дані та мітку до відповідних списків, якщо кількість даних для класу ще не досягла 50К - присвоюємо мітку 0, якщо більше 50К - присвоюємо мітку 1.        
        print(f"Завантажено {len(X)} зразків. Клас <=50K: {count_class1}, Клас >50K: {count_class2}")
        return np.array(X), np.array(y)
    except FileNotFoundError:
        print("Помилка: Файл 'income_data.txt' не знайдено.")
        exit()
#перевірка наявності файлу та виведення повідомлення про помилку, якщо файл не знайдено 


def encode_data(X):
    label_encoders = []
    X_encoded = np.empty(X.shape)

    for i, item in enumerate(X[0]):
        if item.isdigit():
            X_encoded[:, i] = X[:, i]
        else:
            label_encoder = preprocessing.LabelEncoder()
            X_encoded[:, i] = label_encoder.fit_transform(X[:, i])
            label_encoders.append(label_encoder)

    print("Дані успішно закодовані.")
    return X_encoded.astype(float)
#Перетворення категоріальних даних в числові за допомогою LabelEncoder та перетворення масиву даних в тип float

def train_and_evaluate(X_train, X_test, y_train, y_test, kernel_type):
    print(f"\nПочинається навчання з ядром '{kernel_type}'...")  # Додаємо повідомлення перед навчанням
    model = SVC(kernel=kernel_type)
    model.fit(X_train, y_train)
    print(f"Навчання з ядром '{kernel_type}' завершено.")  
    y_pred = model.predict(X_test)
    # Розділення данних на навчальний та тестовий набори,початокнавчання та виведення повідомлення про завершення навчання моделі
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
#Оцінка моделі за допомогою метрик точності, точність серед позитвних результаів, повнота та F1-міра
    
    print(f"\n=== Ядро: {kernel_type} ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
#Виведення результатів оцінки моделі
    
    return accuracy, precision, recall, f1
#Повернення результатів оцінки моделі

X, y = load_data('income_data.txt')
X = encode_data(X)
# Завантаження та кодування даних

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(f"Дані успішно розділено на тренувальні та тестові набори.\nТренувальний набір: {len(X_train)} зразків\nТестовий набір: {len(X_test)} зразків")
# Розділення на навчальний та тестовий набір

kernels = ['poly', 'rbf', 'sigmoid']
X_train = X_train[:10000]
y_train = y_train[:10000]
results = {}
# Оголошення ядер для перевірки та обмеження розміру навчального набору до 10000 зразків 


for kernel in kernels:
    results[kernel] = train_and_evaluate(X_train, X_test, y_train, y_test, kernel)
# Навчання та оцінка для кожного ядра 

print("\nПеревірка: Моделі навчаються успішно.")

for kernel in kernels:
    if kernel in results:
        print(f"Ядро '{kernel}' — навчання завершено успішно.")
    else:
        print(f"Ядро '{kernel}' — не вдалося виконати навчання.")

# Перевірка навчання для кожного ядра