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

def preparar_serie_diaria(df_item):
    # Agrupar ventas por día
    df_agg = df_item.groupby('FECHA_VENTA').agg({
        'CANTIDAD_VENDIDA': 'sum',
        'DESCRIPCION': 'first',
        'ITEM': 'first'
    }).reset_index()

    # Reindexar a frecuencia diaria
    fecha_inicio = df_agg['FECHA_VENTA'].min()
    fecha_fin = df_agg['FECHA_VENTA'].max()
    fechas_diarias = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
    df_diario = pd.DataFrame({'FECHA_VENTA': fechas_diarias})
    df_merged = pd.merge(df_diario, df_agg, on='FECHA_VENTA', how='left')
    df_merged['CANTIDAD_VENDIDA'] = df_merged['CANTIDAD_VENDIDA'].fillna(0)

    # Rellenar columnas faltantes
    df_merged['DESCRIPCION'] = df_merged['DESCRIPCION'].fillna(method='ffill')
    df_merged['ITEM'] = df_merged['ITEM'].fillna(method='ffill')
    
    return df_merged

def entrenar_prophet(df, periodo):
    df_p = df[['FECHA_VENTA', 'CANTIDAD_VENDIDA']].rename(columns={'FECHA_VENTA': 'ds', 'CANTIDAD_VENDIDA': 'y'})
    model = Prophet(weekly_seasonality=True, daily_seasonality=False)
    model.fit(df_p)
    future = model.make_future_dataframe(periods=periodo, freq='D')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

# -----------------------
# Interfaz Streamlit
# -----------------------

st.title("Predicción de Demanda con Prophet")

excel_path = "Items_Morante.xlsx"
df = cargar_datos(excel_path)
items = df['ITEM'].unique()

item_seleccionado = st.selectbox("Selecciona un ítem para analizar:", items)

df_item_raw = df[df['ITEM'] == item_seleccionado].copy()
descripcion = df_item_raw['DESCRIPCION'].iloc[0]
st.write(f"**Descripción del ítem:** {descripcion}")

# Preprocesar serie diaria
df_item = preparar_serie_diaria(df_item_raw)

# Slider para elegir días a predecir
periodo = st.slider("Selecciona el número de días a predecir:", min_value=7, max_value=90, value=45)

# Entrenar modelo
forecast = entrenar_prophet(df_item, periodo)

# Preparar series
df_real = df_item[['FECHA_VENTA', 'CANTIDAD_VENDIDA']].rename(columns={'FECHA_VENTA': 'ds', 'CANTIDAD_VENDIDA': 'y'})
df_comparacion = pd.merge(df_real, forecast, on='ds', how='left')

# Determinar fecha de corte (última fecha real)
fecha_corte = df_real['ds'].max()

# -----------------------
# Gráfico
# -----------------------

fig, ax = plt.subplots(figsize=(14, 6))

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

forecast_futuro = forecast[forecast['ds'] > fecha_corte]
total_predicho = forecast_futuro['yhat'].sum()

st.subheader(f"Total estimado para los próximos {periodo} días:")
st.write(f"**{total_predicho:.0f} unidades estimadas** para importar en {periodo} días.")

# -----------------------
# Métricas de evaluación
# -----------------------

df_eval = df_comparacion.dropna()

if df_eval.empty:
    st.warning("No hay suficientes datos reales para calcular métricas.")
else:
    y_true = df_eval['y']
    y_pred = df_eval['yhat']

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")
