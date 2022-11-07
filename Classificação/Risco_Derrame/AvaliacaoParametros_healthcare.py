
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

with open('Risco_derrame.pkl', mode="rb") as f:
    x_treino, y_treino, x_teste, y_teste = pickle.load(f)

x = np.concatenate([x_treino, x_teste], axis=0)
y = np.concatenate([y_treino, y_teste], axis=0)

with open('AvaliacaoParametros.txt', 'w') as arquivo:
    # Ajuste de parametros algoritmos
    parametros_RF = {'n_estimators': [10, 40, 60, 100, 140],
                     'criterion': ["gini", "entropy"],
                     'min_samples_split': [1, 2, 5],
                     'min_samples_leaf': [1, 3, 6]}

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(), param_grid=parametros_RF)
    grid_search.fit(x, y)

    arquivo.write(
        f"Parâmetros RF: {grid_search.best_params_}\n    Accuracy:  {grid_search.best_score_}\n\n")

    parametros_KNN = {'n_neighbors': [1, 5, 10],
                      'weights': ['uniform', 'distance'],
                      'p': [1, 2, 5]}

    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(), param_grid=parametros_KNN)
    grid_search.fit(x, y)
    arquivo.write(
        f"Parâmetros KNN: {grid_search.best_params_}\n   Accuracy:  {grid_search.best_score_}\n\n")

    parametros_SVM = {'tol': [0.001, 0.0001],
                      'C': [1.0, 2.0],
                      'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

    grid_search = GridSearchCV(
        estimator=SVC(), param_grid=parametros_SVM)
    grid_search.fit(x, y)

    arquivo.write(
        f"Parâmetros SVM: {grid_search.best_params_}\n      Accuracy:  {grid_search.best_score_}\n\n")

    parametros_MLP = {'activation': ['relu', 'logistic', 'tahn'],
                      'solver': ['adam', 'sgd'],
                      'batch_size': [10, 56]}

    grid_search = GridSearchCV(estimator=MLPClassifier(),
                               param_grid=parametros_MLP)
    grid_search.fit(x, y)

    arquivo.write(
        f"Parâmetros MLP: {grid_search.best_params_}\n    Accuracy:  {grid_search.best_score_}\n\n")
