import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------
# Funciones
# -----------------------

@st.cache_data
def cargar_datos(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    df['FECHA_VENTA'] = pd.to_datetime(df['FECHA_VENTA'])
    df = df.sort_values(by='FECHA_VENTA')
    return df

def entrenar_rnn(df, periodo, look_back=5):
    ventas = df['CANTIDAD_VENDIDA'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    ventas_norm = scaler.fit_transform(ventas)

    X, y = [], []
    for i in range(len(ventas_norm) - look_back):
        X.append(ventas_norm[i:i+look_back])
        y.append(ventas_norm[i+look_back])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=20, batch_size=1, verbose=0)

    ultimos = X[-1]
    predicciones = []
    for _ in range(periodo):
        pred = model.predict(ultimos.reshape(1, look_back, 1), verbose=0)
        predicciones.append(pred[0, 0])
        ultimos = np.append(ultimos[1:], [[pred[0, 0]]], axis=0)

    pred_final = scaler.inverse_transform(np.array(predicciones).reshape(-1, 1)).flatten()
    fechas = pd.date_range(start=df['FECHA_VENTA'].max() + pd.Timedelta(days=1), periods=periodo)
    return pd.Series(pred_final, index=fechas)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

# -----------------------
# Interfaz Streamlit
# -----------------------

st.title("Predicción de Demanda con RNN")

excel_path = "Items_Morante.xlsx"
df = cargar_datos(excel_path)
items = df['ITEM'].unique()

item_seleccionado = st.selectbox("Selecciona un ítem para analizar:", items)

df_item = df[df['ITEM'] == item_seleccionado].copy()
descripcion = df_item['DESCRIPCION'].iloc[0]
st.write(f"**Descripción del ítem:** {descripcion}")
periodo = st.slider("Días a predecir", min_value=7, max_value=60, value=45)

# Real
real = df_item.set_index('FECHA_VENTA')['CANTIDAD_VENDIDA'][-periodo:]

# Predicción RNN
rnn_pred = entrenar_rnn(df_item, periodo)

# Visualización
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(real.index, real.values, label='Real', marker='o')
ax.plot(rnn_pred.index, rnn_pred.values, label='RNN', marker='d')
ax.set_title(f'Predicción de ventas para {item_seleccionado} con RNN')
ax.legend()
st.pyplot(fig)

# Evaluación
mae = mean_absolute_error(real.values, rnn_pred.values[:len(real)])
rmse = np.sqrt(mean_squared_error(real.values, rnn_pred.values[:len(real)]))
mape = mean_absolute_percentage_error(real.values, rnn_pred.values[:len(real)])

st.subheader("Evaluación del Modelo RNN")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAPE:** {mape:.2f}%")
