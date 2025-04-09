import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#Підключення бібліотек

def load_data(file_path):
    try:
        X = []
        y = []
        count_class1 = 0
        count_class2 = 0
        max_datapoints = 25000
#Завнтаження файлу та обробка данних, створення списків для даних та міток, обмеження кількості даних
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            print(f"Файл завантажений успішно. Кількість рядків: {len(lines)}")
#Відкриття файлу та зчитування рядків, якщо все успішно, у консолі виводиться кількість рядків
            
            for line in lines:
                if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                    break
                if '?' in line:
                    continue

                data = line.strip().split(', ')
#Проходження по рядках, якщо рядок містить '?' - пропускаємо, якщо рядок містить більше 25000 даних - зупиняємо цикл, розбиваємо кожен рядок на список значень (data) 
                
                if data[-1] == '<=50K' and count_class1 < max_datapoints:
                    X.append(data[:-1])
                    y.append(0)
                    count_class1 += 1
                elif data[-1] == '>50K' and count_class2 < max_datapoints:
                    X.append(data[:-1])
                    y.append(1)
                    count_class2 += 1
        
        print(f"Завантажено {len(X)} зразків. Клас <=50K: {count_class1}, Клас >50K: {count_class2}")
#Якщо дохід менше/дорівнює 50К, додаємо мітку 0; якщо більше — мітку 1. Додаємо дані до списків X та y. Збільшуємо лічильники для кожного класу. Виводимо кількість завантажених зразків та кількість зразків для кожного класу.

        return np.array(X), np.array(y)
#Повертаємо X та y як масиви NumPy
    
    except FileNotFoundError:
        print("Помилка: Файл 'income_data.txt' не знайдено.")
        exit()
#Якщо файл не знайдено, виводимо помилку та завершуємо роботу програми


def encode_data(X):
    label_encoders = []
    X_encoded = np.empty(X.shape)
#Створюємо порожній масив для закодованих даних
    
    for i, item in enumerate(X[0]):
        if item.isdigit():
            X_encoded[:, i] = X[:, i]
        else:
            label_encoder = preprocessing.LabelEncoder()
            X_encoded[:, i] = label_encoder.fit_transform(X[:, i])
            label_encoders.append(label_encoder)
    
    print("Дані успішно закодовані.")
#Проходимо по кожному стовпцю даних. Якщо значення є цифровим, додаємо його без змін. Якщо значення є текстом, кодуємо його за допомогою LabelEncoder. Виводимо повідомлення про успішне кодування даних.
    
    return X_encoded.astype(float)
#Повертаємо закодований масив як масив з типом float

def train_and_evaluate(X_train, X_test, y_train, y_test, kernel_type):
    model = SVC(kernel=kernel_type, max_iter=100000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
#Приймаємо тренувальні та тестові дані, тип ядра. Створюємо модель SVM з вказаним ядром, навчаємо модель на тренувальних даних, прогнозуємо мітки для тестового набору даних.
    
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='binary', zero_division='warn')
    recall = metrics.recall_score(y_test, y_pred, average='binary', zero_division='warn')
    f1 = metrics.f1_score(y_test, y_pred, average='binary', zero_division='warn')
#Обчислюємо 4 основні метрии 
    
    print(f"\n=== Ядро: {kernel_type} ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
#Виводимо результати
    
    return accuracy, precision, recall, f1
#Повертаємо значення метрик назад


X, y = load_data('income_data.txt')
X = encode_data(X)
#Завантажуємо та кодуємо набір дані

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(f"Дані успішно розділено на тренувальні та тестові набори.\nТренувальний набір: {len(X_train)} зразків\nТестовий набір: {len(X_test)} зразків")
#Розділяємо дані на тренувальний та тестовий набори, виводимо кількість зразків у кожному наборі

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#Масштабуємо дані за допомогою StandardScaler

kernels = ['poly', 'rbf', 'sigmoid']
X_train = X_train[:10000]  # Обмеження тренувального набору для швидшого навчання
y_train = y_train[:10000]
results = {}
#Визначаємо типи ядер, обмежуємо тренувальний набір, створюємо словник для збереження результатів

for kernel in kernels:
    results[kernel] = train_and_evaluate(X_train, X_test, y_train, y_test, kernel)
#Навчаємо та оцінюємо модель для кожного типу ядра, зберігаємо результати у словник

fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors = ['blue', 'green', 'red', 'purple']
#Створюємо простір для графіку, визначаємо назви метрик та кольори 

for i, metric_name in enumerate(metrics_names):
    values = [results[kernel][i] for kernel in kernels]
    ax.plot(kernels, values, label=metric_name, marker='o', color=colors[i])
#Проходимо по кожній метриці, витягуємо значення для кожного ядра, створюємо графік

ax.set_title('Порівняння продуктивності різних SVM ядер на даних про доходи')
ax.set_xlabel('Тип ядра')
ax.set_ylabel('Значення метрики')
ax.legend()
plt.show()
#Налаштовуємо заголовок, мітки осей, легенду та відображаємо графік