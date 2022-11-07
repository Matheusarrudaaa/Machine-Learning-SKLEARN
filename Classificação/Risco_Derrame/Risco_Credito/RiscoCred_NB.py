import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

#Recebendo os atributos
prev = [];
while True:
    atr = input("Histórico: Bom / Ruim / Desconhecido\n$ ")
    if atr == 'bom' or atr == 'Bom':
        prev.append(0)
        break;
    if atr == 'ruim' or atr == 'Ruim':
        prev.append(2)
        break;
    if atr == 'Desconhecido' or atr == 'desconhecido':
        prev.append(1)
        break;
    else:
        print("Atributo inválido")

while True:
    atr = input("Dívida: Alta / Baixa\n$ ")
    if atr == 'alta' or atr == 'Alta':
        prev.append(0)
        break;
    if atr == 'Baixa' or atr == 'baixa':
        prev.append(1)
        break;
    else:
        print("Atributo inválido")

while True:
    atr = input("Garantias: Nenhuma / Adequada\n$ ")
    if atr == 'nenhuma' or atr == 'Nenhuma':
        prev.append(1)
        break;
    if atr == 'Adequada' or atr == 'adequada':
        prev.append(0)
        break;
    else:
        print("Atributo inválido")

while True:
    atr = input("Renda: 0_15 / 15_35 / 35_\n$ ")
    if atr == '0_15':
        prev.append(0)
        break;
    if atr == '15_35':
        prev.append(1)
        break;
    if atr == '35_':
        prev.append(2)
        break;
    else:
        print("Atributo inválido")


risco_cred = pd.read_csv('./Banco_de_dados/risco_credito.csv');

#criação de previsores e classe
x_risco_cred = risco_cred.iloc[:, 0:4].values;
y_risco_cred = risco_cred.iloc[:,4].values;

#historia divida garantias     renda     risco
label_encoder_historia = LabelEncoder();
label_encoder_divida = LabelEncoder();
label_encoder_garantias = LabelEncoder();
label_encoder_renda = LabelEncoder();

#Padronização dos dados
x_risco_cred[:,0] = label_encoder_historia.fit_transform(x_risco_cred[:,0])
x_risco_cred[:,1] = label_encoder_divida.fit_transform(x_risco_cred[:,1])
x_risco_cred[:,2] = label_encoder_garantias.fit_transform(x_risco_cred[:,2])
x_risco_cred[:,3] = label_encoder_renda.fit_transform(x_risco_cred[:,3])

"""
Pela padronização, os valores gerados:
historia- ruim(2), desconhecida(1), boa(0)
divida-   alta(0), baixa(1)
garantias- nenhuma(1), adequada(0)
renda- 0_15(0), 15_35(1), acima_35(2)


ex: historia boa, divida alta, garantias nenhuma, renda acima_35
            0          0           1                2
"""

#Algoritmo que irá treinar e gerar a tabela de probabilidades
naive_risk = GaussianNB();
naive_risk.fit(x_risco_cred, y_risco_cred);

#Gerar previsão
previsao = naive_risk.predict([prev])
print(f"\nRisco de Empréstimo: {previsao[0].title()}\n")

