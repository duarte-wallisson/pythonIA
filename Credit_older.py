import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

base_credit = pd.read_csv('content/credit_data.csv')

# clientId == nominal, age == contínua
# 0 == pagou, 1 == não pagou

# Correções
# valores negativos
base_credit.loc[base_credit['age'] < 0, 'age'] = base_credit['age'][base_credit['age'] > 0].mean()
# valores nulos
base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)

# Aula 16
# print(base_credit)
# print(base_credit.head())
# print(base_credit.tail())
# print(base_credit.describe())
# print(base_credit[base_credit['income'] >= 69995.685578])

# Aula 17
# print(np.unique(base_credit['default'], return_counts=True))
# print(sns.countplot(x=base_credit['default']))

# plt.hist(x=base_credit['age'])
# plt.hist(x=base_credit['income'])
# plt.hist(x=base_credit['loan'])
# plt.show()
# grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
# grafico.show()

# Aula 18
# print(base_credit.loc[base_credit['age'] < 0])  # print(base_credit[base_credit['age'] < 0])
# --Apagar a coluna inteira (de todos os registros da base de dados)
# base_credit2 = base_credit.drop('age', axis = 1)
# print(base_credit2)
# --Apagar somente os registros com valores inconsistentes
# base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
# base_credit3
# --Preencher os valores inconsistente manualmente
# --Prencher a média
# print(base_credit['age'].mean())
# print(base_credit['age'][base_credit['age'] > 0].mean()) #--Necessário
# base_credit.loc[base_credit['age'] < 0, 'age'] = base_credit['age'][base_credit['age'] > 0].mean()
# Obs.: Necessário o parâmetro 2, senão atualiza todas as colunas
# print(base_credit['age'].mean())
# print(base_credit.head(27))

# Aula 19
# print(base_credit.isnull().sum())
# print(base_credit.loc[pd.isnull(base_credit['age'])]) #--Necessário
# base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)
# print(base_credit.loc[pd.isnull(base_credit['age'])])
# print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])

# # Aula 20 - Alocação dos previsores e a classe
# print(type(base_credit))
# # -- Aloca os previsores, exceto o id
X_credit = base_credit.iloc[:, 1:4].values
# print(type(X_credit))
# # -- Aloca a classe
y_credit = base_credit.iloc[:, 4].values
# print(type(y_credit))

# Aula 21 - Escalonamento
# print(X_credit)
# print(X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min())
# print(X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max())
# --Padronização
scaler_credit = StandardScaler()
# --Normalização
# scaler_credit = MinMaxScaler
X_credit = scaler_credit.fit_transform(X_credit)
# print(X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min())
# print(X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max())
# print(X_credit)

# Aula 28 e 29
# random_state é necessário para manter os valores
X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(X_credit, y_credit,
                                                                                              test_size=0.25,
                                                                                              random_state=0)

# Aula 30
with open('content/credit_1.pkl', mode='wb') as f:
    pickle.dump([X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste], f)
