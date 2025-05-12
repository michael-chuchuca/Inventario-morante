import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

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

def entrenar_prophet(df, periodo):
    df_p = df[['FECHA_VENTA', 'CANTIDAD_VENDIDA']].rename(columns={'FECHA_VENTA': 'ds', 'CANTIDAD_VENDIDA': 'y'})
    model = Prophet()
    model.fit(df_p)
    future = model.make_future_dataframe(periods=periodo)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].set_index('ds')

# -----------------------
# Interfaz Streamlit
# -----------------------

st.title("Predicción de Demanda con Prophet")

excel_path = "Items_Morante.xlsx"
df = cargar_datos(excel_path)
items = df['ITEM'].unique()

item_seleccionado = st.selectbox("Selecciona un ítem para analizar:", items)

df_item = df[df['ITEM'] == item_seleccionado].copy()
descripcion = df_item['DESCRIPCION'].iloc[0]
st.write(f"**Descripción del ítem:** {descripcion}")
periodo = 1  # Solo se predice 1 día para comparar con el último real

# Datos reales
real = df_item.set_index('FECHA_VENTA')['CANTIDAD_VENDIDA']
fecha_real_final = real.index[-1]
valor_real_final = real.values[-1]

# Predicción con Prophet
prophet_pred = entrenar_prophet(df_item, periodo)
fecha_pred = prophet_pred.index[-1]
valor_pred = prophet_pred['yhat'].values[-1]

# Gráfico
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(real.index, real.values, label='Serie Real', alpha=0.3, color='gray')
ax.scatter(fecha_real_final, valor_real_final, color='blue', label='Real')
ax.scatter(fecha_pred, valor_pred, color='green', label='Predicción Prophet')

# Etiquetas
ax.annotate(f'Real: {valor_real_final:.0f}', (fecha_real_final, valor_real_final), 
            textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='blue')
ax.annotate(f'Predicción: {valor_pred:.0f}', (fecha_pred, valor_pred), 
            textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, color='green')

ax.set_title(f'Última predicción para {item_seleccionado} con Prophet')
ax.legend()
st.pyplot(fig)

# Evaluación (solo si se compara con valor real conocido)
st.subheader("Evaluación de la Predicción (último punto)")
mae = mean_absolute_error([valor_real_final], [valor_pred])
rmse = np.sqrt(mean_squared_error([valor_real_final], [valor_pred]))
mape = np.mean(np.abs((valor_real_final - valor_pred) / (valor_real_final + 1e-10))) * 100

st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAPE:** {mape:.2f}%")
