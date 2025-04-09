import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from typing import Any
#Завантаження бібліотек


iris: Any = load_iris()
iris = load_iris()
X, y = iris.data, iris.target
#Завантажуєбо набір даних Iris та розділяємо його на ознаки (X) та мітки (y)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)
#Розділяємо дані на навчальний та тестовий набори у відношенні 70/30

clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(Xtrain, ytrain)
#Створюємо та навчаємо модель RidgeClassifier

ypred = clf.predict(Xtest)
#Передбачає класи для тестового набору даних


print('Accuracy:', np.round(metrics.accuracy_score(ytest, ypred), 4))
print('Precision:', np.round(metrics.precision_score(ytest, ypred, average='weighted'), 4))
print('Recall:', np.round(metrics.recall_score(ytest, ypred, average='weighted'), 4))
print('F1 Score:', np.round(metrics.f1_score(ytest, ypred, average='weighted'), 4))
print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(ytest, ypred), 4))
print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(ytest, ypred), 4))
#Виводимо метрики для оцінки якості моделі


print('\nClassification Report:\n', metrics.classification_report(ytest, ypred))
#Виводимо звіт про класифікацію (precision, recall, f1-score та support для кожного класу)

mat = confusion_matrix(ytest, ypred)
#створюємо матрицю помилок, яка показує кількість правильних та неправильних передбачень для кожного класу

sns.set_theme()
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title('Confusion Matrix — RidgeClassifier (Iris)')
plt.savefig("Confusion.jpg")
#Візуалізуємо матрицю помилок за допомогою теплової карти, зберігаємо зображення у форматі JPG

