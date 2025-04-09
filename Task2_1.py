import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#іморт необхідних бібліотек для роботи з даними та навчанням моделі

input_file = 'income_data.txt'
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000
#завнтажуємо необхідний файл, створюємо пусті списки для даних та міток, задаємо лічильники для класів та максимальну кількість даних

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            y.append(0)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            y.append(1)
            count_class2 += 1

#Тут ми зчитуємо дані рядок за рядком, пропускаємо рядки з символом '?', розділяємо рядок на дані та мітку, та додаємо їх до відповідних списків, якщо "<=50к" - клас 0, ">50к" - клас 1

X = np.array(X)
label_encoders = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_enc = preprocessing.LabelEncoder()
        X_encoded[:, i] = label_enc.fit_transform(X[:, i])
        label_encoders.append(label_enc)

X = X_encoded[:, :-1].astype(int)
y = np.array(y)
# Тут ми перетворюємо дані в масив numpy, створюємо список для кодування міток, та створюємо новий масив для закодованих даних. Якщо елемент є цифрою, ми просто додаємо його до нового масиву, якщо ні, ми використовуємо LabelEncoder для кодування міток та додаємо його до списку кодування міток

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)
# Розділяємо дані на навчальну та тестову вибірки в співвідношенні 80/20, створюємо класифікатор OneVsOne з лінійним SVM та навчаємо його на навчальній вибірці

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# Прогнозуємо мітки для тестової вибірки, обчислюємо точність, точність, повноту та F1-міру

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
# Виводимо результати на екран