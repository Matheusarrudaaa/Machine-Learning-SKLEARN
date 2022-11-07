import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

with open('Risco_derrame.pkl', mode="rb") as f:
    x_treino, y_treino, x_teste, y_teste = pickle.load(f)

model = SVC(C=1.0, kernel='rbf', tol=0.001)
model.fit(x_treino, y_treino)

prev = model.predict(x_teste)
print(accuracy_score(y_teste, prev))
print(classification_report(y_teste, prev))
