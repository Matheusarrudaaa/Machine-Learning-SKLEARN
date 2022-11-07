import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

with open("./House_Prices.pkl", mode="rb") as f:
    X_treino, X_teste, Y_treino, Y_teste = pickle.load(f)

#Treinando o Modelo
model = RandomForestRegressor(random_state=26)
model.fit(X_treino, Y_treino)

prev = model.predict(X_teste)

#Pontuação
print(r2_score(Y_teste, prev))

#Erro
print(mean_squared_error(Y_teste, prev))
print(mean_absolute_error(Y_teste, prev))