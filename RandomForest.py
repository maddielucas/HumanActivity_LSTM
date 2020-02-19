import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

names = ['Window No', 'Avg of x', 'Avg of y', 'Avg of z', 'Min of x', 'Min of y', 'Max of x', 'Max of y', 'Max of z',
         'Variance', 'Class']
dataset = pd.read_csv("Laundry.csv", names=names, encoding='ascii')

print(dataset.head())
x = dataset.drop('Class', axis=1)
y = dataset.Class


X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.5, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier(n_estimators=1, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
