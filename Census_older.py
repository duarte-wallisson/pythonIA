import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split

# Abre um gráfico com mais opções
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

# Aula 22 e 23
base_census = pd.read_csv('content/census.csv')
# print(base_census.describe())
# print(base_census['age'].isnull().sum())
# plt.hist(x = base_census['hour-per-week']);
# plt.show()
# grafico = px.treemap(base_census, path=['workclass', 'age'])
# grafico = px.scatter_matrix(base_census, dimensions=['age', 'workclass'])
# grafico = px.parallel_categories(base_census, dimensions=['workclass', 'occupation', 'income'])
# grafico.show()

# Aula 24
X_census = base_census.iloc[:, 0:14].values
# print(X_census)
y_census = base_census.iloc[:, 14].values
# print(y_census)

# Aula 25
# Ruim, pois transforma em números inteiros em sequência, logo a IA pode dar um peso maior
# label_encoder_workclass = LabelEncoder()
# label_encoder_education = LabelEncoder()
# label_encoder_marital = LabelEncoder()
# label_encoder_occupation = LabelEncoder()
# label_encoder_relationship = LabelEncoder()
# label_encoder_race = LabelEncoder()
# label_encoder_sex = LabelEncoder()
# label_encoder_country = LabelEncoder()
#
# X_census[:, 1] = label_encoder_workclass.fit_transform(X_census[:, 1])
# X_census[:, 3] = label_encoder_education.fit_transform(X_census[:, 3])
# X_census[:, 5] = label_encoder_marital.fit_transform(X_census[:, 5])
# X_census[:, 6] = label_encoder_occupation.fit_transform(X_census[:, 6])
# X_census[:, 7] = label_encoder_relationship.fit_transform(X_census[:, 7])
# X_census[:, 8] = label_encoder_race.fit_transform(X_census[:, 8])
# X_census[:, 9] = label_encoder_sex.fit_transform(X_census[:, 9])
# X_census[:, 13] = label_encoder_country.fit_transform(X_census[:, 13])

# print(X_census)

# Aula 26
# print(len(np.unique(base_census['workclass'])))
# print(X_census.shape)
# Codifica os valores de modo a não colocar pesos
# remainder='passthrough' impede de remover as outras colunas
onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],
                                         remainder='passthrough')
X_census = onehotencoder_census.fit_transform(X_census).toarray()
# print(X_census.shape)
# print(len(np.unique(base_census['workclass'])))

# Aula 27
scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)

# Aula 28 e 29
# random_state é necessário para manter os valores
X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = train_test_split(X_census, y_census,
                                                                                              test_size=0.15,
                                                                                              random_state=0)
# print(X_census_teste.shape, y_census_teste.shape)

# Aula 30
with open('content/census.pkl', mode='wb') as f:
    pickle.dump([X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste], f)
