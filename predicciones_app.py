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
    return forecast[['ds', 'yhat']]

# -----------------------
# Interfaz Streamlit
# -----------------------

st.title("📈 Predicción de Demanda con Prophet")

excel_path = "Items_Morante.xlsx"
df = cargar_datos(excel_path)
items = df['ITEM'].unique()

item_seleccionado = st.selectbox("Selecciona un ítem para analizar:", items)

df_item = df[df['ITEM'] == item_seleccionado].copy()
descripcion = df_item['DESCRIPCION'].iloc[0]
st.write(f"**Descripción del ítem:** {descripcion}")

# Slider para elegir días a predecir
periodo = st.slider("Selecciona el número de días a predecir:", min_value=7, max_value=90, value=45)

# Entrenar modelo
forecast = entrenar_prophet(df_item, periodo)

# Preparar series
df_real = df_item[['FECHA_VENTA', 'CANTIDAD_VENDIDA']].rename(columns={'FECHA_VENTA': 'ds', 'CANTIDAD_VENDIDA': 'y'})
df_comparacion = pd.merge(df_real, forecast, on='ds', how='left')

# Separar futuro
fecha_corte = df_real['ds'].max()
forecast_futuro = forecast[forecast['ds'] > fecha_corte]
total_predicho = forecast_futuro['yhat'].sum()

# -----------------------
# Gráfico
# -----------------------

fig, ax = plt.subplots(figsize=(14, 6))  # Más ancho

# Línea azul: valores reales
ax.plot(df_comparacion['ds'], df_comparacion['y'], 'b--', label='Cantidad Real', linewidth=2)

# Línea roja: valores pronosticados (histórico + futuro)
ax.plot(forecast['ds'], forecast['yhat'], 'r--', label='Cantidad Pronosticada', linewidth=2)

# Línea vertical para indicar el inicio del pronóstico
ax.axvline(fecha_corte, color='gray', linestyle=':', alpha=0.7)
ax.annotate('Inicio de Predicción', xy=(fecha_corte, ax.get_ylim()[1]*0.9),
            xytext=(10, 0), textcoords='offset points', fontsize=10, color='gray')

# Título y etiquetas
ax.set_title("Pronóstico de Ventas con Valores Reales", fontsize=15)
ax.set_xlabel("Fecha", fontsize=12)
ax.set_ylabel("Cantidad Vendida", fontsize=12)

# Estética
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)

st.pyplot(fig)

# -----------------------
# Total estimado a futuro
# -----------------------
st.subheader("Evaluación del Pronóstico desde el Inicio de la Predicción")

# Filtrar los datos comparables desde el inicio de la predicción
df_eval = df_comparacion[df_comparacion['ds'] >= fecha_corte].copy()
df_eval = df_eval.dropna()  # Asegura que no haya nulos en y o yhat

# Valores reales y pronosticados
y_true = df_eval['y']
y_pred = df_eval['yhat']

# Métricas
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

# Mostrar
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAPE:** {mape:.2f}%")
