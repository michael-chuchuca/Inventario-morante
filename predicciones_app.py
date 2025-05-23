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

def preparar_serie_semanal(df_item_raw):
    df_agg = df_item_raw.groupby(pd.Grouper(key='FECHA_VENTA', freq='W')).agg({
        'CANTIDAD_VENDIDA': 'sum',
        'DESCRIPCION': 'first',
        'ITEM': 'first'
    }).reset_index()
    df_agg = df_agg.rename(columns={'FECHA_VENTA': 'ds', 'CANTIDAD_VENDIDA': 'y'})
    return df_agg

def entrenar_prophet_semanal(df, periodo_semanas):
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periodo_semanas, freq='W')
    forecast = model.predict(future)
    # Evitar predicciones negativas
    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))
    return forecast[['ds', 'yhat']]

# -----------------------
# Interfaz Streamlit
# -----------------------

st.title("Predicción de Demanda Semanal con Prophet")

excel_path = "Items_Morante.xlsx"
df = cargar_datos(excel_path)
items = df['ITEM'].unique()

item_seleccionado = st.selectbox("Selecciona un ítem para analizar:", items)

df_item_raw = df[df['ITEM'] == item_seleccionado].copy()
descripcion = df_item_raw['DESCRIPCION'].iloc[0]
st.write(f"**Descripción del ítem:** {descripcion}")

# Slider para elegir días a predecir
periodo_dias = st.slider("Selecciona el número de días a predecir:", min_value=7, max_value=90, value=45)

# Convertir días a semanas (redondeando hacia arriba)
periodo_semanas = int(np.ceil(periodo_dias / 7))

# Preparar serie semanal
df_semanal = preparar_serie_semanal(df_item_raw)

# Entrenar modelo
forecast = entrenar_prophet_semanal(df_semanal, periodo_semanas)

# Preparar datos reales para comparar
df_real = df_semanal.copy()

# Merge para comparación
df_comparacion = pd.merge(df_real, forecast, on='ds', how='left')

# Última fecha real
fecha_corte = df_real['ds'].max()

# -----------------------
# Gráfico
# -----------------------

fig, ax = plt.subplots(figsize=(14, 6))

# Línea azul: valores reales
ax.plot(df_comparacion['ds'], df_comparacion['y'], 'b--', label='Cantidad Real', linewidth=2)

# Línea roja: valores pronosticados (histórico + futuro)
ax.plot(forecast['ds'], forecast['yhat'], 'r--', label='Cantidad Pronosticada', linewidth=2)

# Línea vertical para indicar inicio de la predicción
ax.axvline(fecha_corte, color='gray', linestyle=':', alpha=0.7)
ax.annotate('Inicio de Predicción', xy=(fecha_corte, ax.get_ylim()[1]*0.9),
            xytext=(10, 0), textcoords='offset points', fontsize=10, color='gray')

ax.set_title("Pronóstico Semanal de Ventas con Valores Reales", fontsize=15)
ax.set_xlabel("Fecha", fontsize=12)
ax.set_ylabel("Cantidad Vendida (semanal)", fontsize=12)
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)

st.pyplot(fig)

# -----------------------
# Total estimado a futuro (en unidades, aproximado diario)
# -----------------------

forecast_futuro = forecast[forecast['ds'] > fecha_corte].copy()
if forecast_futuro.empty:
    total_predicho = 0
else:
    total_predicho = forecast_futuro['yhat'].sum()
    
# Convertir total semanal a unidades diarias estimadas
# Asumiendo uniformidad, multiplicamos por 7 (días/semana)
total_diario_estimado = total_predicho * (periodo_dias / (periodo_semanas * 7))

st.subheader(f"Total estimado para los próximos {periodo_dias} días:")
st.write(f"**{total_diario_estimado:.0f} unidades estimadas** para importar en {periodo_dias} días.")

# -----------------------
# Métricas de evaluación
# -----------------------

# Solo evaluar en el rango histórico
df_eval = df_comparacion.dropna().copy()
df_eval = df_eval[df_eval['y'] > 0]  # Evitar división por cero en MAPE

if df_eval.empty:
    st.warning("No hay suficientes datos reales > 0 para calcular métricas.")
else:
    y_true = df_eval['y']
    y_pred = df_eval['yhat']

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")
