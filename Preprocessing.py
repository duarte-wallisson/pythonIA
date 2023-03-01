import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def pre_processing(path, save, begin):
    base = pd.read_csv(path)
    base = treat_values(base)
    X = extract_X(base, -1, begin)
    y = extract_y(base, -1)
    X = one_hot_encoder(X)
    X = scale_data(X)
    split = split_data(X, y)
    write_pkl(save, *split)


def treat_values(base):
    for column in base.columns:
        if pd.api.types.is_numeric_dtype(base[column]):
            # valores negativos
            base.loc[base[column] < 0, column] = base[column][base[column] > 0].mean()
            # valores nulos
            base[column].fillna(base[column].mean(), inplace=True)
    return base


def extract_X(base, size, begin):
    # -- Aloca os previsores, exceto o id
    return base.iloc[:, begin:size].values
    # print(type(X_credit))


def extract_y(base, size):
    # -- Aloca a classe
    return base.iloc[:, size].values


def one_hot_encoder(X):
    non_numeric_cols = np.where(np.array([type(X[0, i]) for i in range(X.shape[1])]) == np.dtype('str'))[0]
    if len(non_numeric_cols) > 0:
        onehotencoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), non_numeric_cols)],
                                          remainder='passthrough')
        X = onehotencoder.fit_transform(X).toarray()
    return X


def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# divide em treinamento e teste
def split_data(X, y, test_size=0.25, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, y_train, X_test, y_test


def write_pkl(locale, *split):
    with open(locale, mode='wb') as f:
        pickle.dump(split, f)
