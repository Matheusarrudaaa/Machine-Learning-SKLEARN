import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

db = pd.read_csv("./kc_house_data.csv")

print(db.corr(numeric_only=True)['price'].sort_values()) #Ver a correlação entre os atributos e o preço

#previsores e classe
Y = db.iloc[:, 2].values
X = db.drop([ 'zipcode', 'id', 'date', 'long', 'condition', 'yr_built', 'sqft_lot15', 'sqft_lot', 'yr_renovated', 'price'], axis=1).values #Exclusão de atributos com baixa correlação com preço


#Padronização
norm_x = MinMaxScaler()
norm_y = MinMaxScaler()

X_norm = norm_x.fit_transform(X)
Y_norm = norm_y.fit_transform(Y.reshape(-1, 1))

#Separação teste e treino
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.20, random_state=0)

print(X_treino.shape)
with open("./House_Prices.pkl", mode= "wb") as f:
    pickle.dump([X_treino, X_teste, Y_treino, Y_teste], f)