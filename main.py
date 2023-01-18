import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

base_credit = pd.read_csv('content/credit_data.csv')

# clientId == nominal, age == contínua
# 0 == pagou, 1 == não pagou

# Aula 17
# print(base_credit)
# print(base_credit.head())
# print(base_credit.tail())
# print(base_credit.describe())
# print(base_credit[base_credit['income'] >= 69995.685578])

# Aula 18
# print(np.unique(base_credit['default'], return_counts=True))
# print(sns.countplot(x=base_credit['default']))

# sns.countplot(x=base_credit['default'])
# plt.hist(x=base_credit['age'])
# plt.hist(x=base_credit['income'])
# plt.hist(x=base_credit['loan'])
# plt.show()
grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
grafico.show()
