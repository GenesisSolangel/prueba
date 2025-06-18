import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime
import os

# 1. FUNCION DE IMPUTACION ESTACIONAL

def imputar_datos_estacionales(df):
    """
    Imputa valores NaN en la columna 'value' basándose en la media por hora y día de la semana.
    Si no hay NaNs, retorna el mismo DataFrame.
    """
    if df['value'].isna().sum() == 0:
        print(" No se encontraron valores NaN en 'value'.")
        return df

    print(" Se encontraron valores NaN. Imputando...")

    df['weekday'] = df['datetime'].dt.dayofweek  # Lunes=0, Domingo=6
    df['hour'] = df['datetime'].dt.hour

    media_estacional = (
        df.groupby(['weekday', 'hour'])['value']
        .mean()
        .reset_index()
        .rename(columns={'value': 'media_estacional'})
    )

    df = df.merge(media_estacional, on=['weekday', 'hour'], how='left')
    df['value'] = df['value'].fillna(df['media_estacional'])
    df = df.drop(columns=['media_estacional'])

    print(" Imputación completada usando estacionalidad.")
    return df

# 2. FUNCION DE ESCALADO Y GUARDADO DEL SCALER
def escalar_datos(df, save_path='scaler.pkl'):
    scaler = MinMaxScaler()
    df['value_scaled'] = scaler.fit_transform(df[['value']])
    with open(save_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f" Datos escalados y scaler guardado en '{save_path}'.")
    return df, scaler

# 3. FUNCION PARA DIVIDIR EN TRAIN Y TEST
def dividir_train_test(df, test_size=0.2):
    df = df.copy()
    df = df.set_index('datetime')
    split_point = int(len(df) * (1 - test_size))
    df_train = df.iloc[:split_point]
    df_test = df.iloc[split_point:]
    return df_train, df_test

