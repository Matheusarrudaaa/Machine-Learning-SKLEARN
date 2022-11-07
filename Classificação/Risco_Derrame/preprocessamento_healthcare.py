import pickle

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

data_base = pd.read_csv("./healthcare-dataset-stroke-data.csv")
#print(len(data_base['smoking_status'] == 'Unknown'))

# Pre-Processamento

# Dados Faltantes e incosistentes

data_base.isnull().sum()
data_base['bmi'].fillna(data_base['bmi'].mean(), inplace=True)
data_base = data_base.drop(data_base[data_base['gender'] == 'Other']. index)

# Escalonamento

"""
['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status', 'stroke']
"""

# Separação entre previsores e classe
x = data_base.iloc[:, 1:11].values
y = data_base.iloc[:, 11].values

# Atributos categóricos (LabelEncoder)

labelenc_gender = LabelEncoder()
labelenc_ever_married = LabelEncoder()
lebelenc_work_type = LabelEncoder()
lebelenc_Residence_type = LabelEncoder()
lebelenc_smoking_status = LabelEncoder()

x[:, 0] = labelenc_gender.fit_transform(x[:, 0])
x[:, 4] = labelenc_ever_married.fit_transform(x[:, 4])
x[:, 5] = lebelenc_work_type.fit_transform(x[:, 5])
x[:, 6] = lebelenc_Residence_type.fit_transform(x[:, 6])
x[:, 9] = lebelenc_smoking_status .fit_transform(x[:, 9])


smote = SMOTE(sampling_strategy="minority")
x, y = smote.fit_resample(x, y)
print(x.shape)


"""# Atributos categóricos (onehotEncoder)
onehot_hds = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [
                               0, 4, 5, 6, 9])], remainder='passthrough')
x = onehot_hds.fit_transform(x)

# Padronização
scaler = StandardScaler()
x = scaler.fit_transform(x)"""


"""# Divisão teste e treino
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.20, random_state=1)

with open('Risco_derrame.pkl', mode='wb') as f:
    pickle.dump([x_treino, y_treino, x_teste, y_teste], f)
"""


"""from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

from sklearn.ensemble import ExtraTreesClassifier

select = ExtraTreesClassifier()
select.fit(x, y)

importancia = select.feature_importances_
print(importancia)"""
