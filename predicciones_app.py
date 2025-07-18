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

    # Rellenar fechas faltantes con 0
    fecha_inicio = df_agg['ds'].min()
    fecha_fin = df_agg['ds'].max()
    todas_las_fechas = pd.date_range(fecha_inicio, fecha_fin, freq='W')
    df_agg = df_agg.set_index('ds').reindex(todas_las_fechas).fillna({'y': 0}).reset_index()
    df_agg = df_agg.rename(columns={'index': 'ds'})

    # 🔹 Eliminar outliers extremos (top 2%)
    umbral_extremo = df_agg['y'].quantile(0.98)
    df_agg['y'] = np.where(df_agg['y'] > umbral_extremo, umbral_extremo, df_agg['y'])

    # 🔹 Suavizado doble + clipping
    df_agg['y'] = df_agg['y'].rolling(window=3, min_periods=1).mean()
    df_agg['y'] = df_agg['y'].clip(upper=df_agg['y'].quantile(0.95))
    df_agg['y'] = df_agg['y'].rolling(window=2, min_periods=1).mean()

    # Recuperar columnas perdidas
    df_agg['DESCRIPCION'] = df_item_raw['DESCRIPCION'].iloc[0]
    df_agg['ITEM'] = df_item_raw['ITEM'].iloc[0]

    return df_agg

def entrenar_prophet_semanal(df, periodo_semanas):
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=1.0,
        changepoint_range=0.9
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=periodo_semanas, freq='W')
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))
    return forecast[['ds', 'yhat']]

# -----------------------
# Interfaz Streamlit
# -----------------------

st.markdown("<h1 style='text-align: center;'>Predicción de Demanda Semanal con Prophet Optimizado</h1>", unsafe_allow_html=True)

excel_path = "Items_Morante.xlsx"
df = cargar_datos(excel_path)

df['ITEM_DESC'] = df['ITEM'].astype(str) + " - " + df['DESCRIPCION'].astype(str)
item_opciones = df[['ITEM', 'ITEM_DESC']].drop_duplicates().set_index('ITEM_DESC')
item_seleccionado_desc = st.selectbox("Selecciona un ítem:", item_opciones.index)
item_seleccionado = item_opciones.loc[item_seleccionado_desc]['ITEM']

df_item_raw = df[df['ITEM'] == item_seleccionado].copy()
descripcion = df_item_raw['DESCRIPCION'].iloc[0]
st.write(f"**Descripción del ítem:** {descripcion}")

periodo_dias = st.slider("Selecciona el número de días a predecir:", min_value=7, max_value=90, value=45)
periodo_semanas = int(np.ceil(periodo_dias / 7))

df_semanal = preparar_serie_semanal(df_item_raw)
forecast = entrenar_prophet_semanal(df_semanal, periodo_semanas)

df_real = df_semanal.copy()
df_comparacion = pd.merge(df_real, forecast, on='ds', how='left')
fecha_corte = df_real['ds'].max()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_comparacion['ds'], df_item_raw.groupby(pd.Grouper(key='FECHA_VENTA', freq='W'))['CANTIDAD_VENDIDA'].sum().reset_index()['CANTIDAD_VENDIDA'], 'c--', label='Serie Original', linewidth=1.2)
ax.plot(df_comparacion['ds'], df_comparacion['y'], 'b-', label='Serie Suavizada', linewidth=2)
ax.plot(forecast['ds'], forecast['yhat'], 'r--', label='Cantidad Pronosticada', linewidth=2)
ax.axvline(fecha_corte, color='gray', linestyle=':', alpha=0.7)
ax.annotate('Inicio de Predicción', xy=(fecha_corte, ax.get_ylim()[1]*0.9),
            xytext=(10, 0), textcoords='offset points', fontsize=10, color='gray')
ax.set_title("Pronóstico Semanal de Ventas con Prophet Optimizado", fontsize=15)
ax.set_xlabel("Fecha", fontsize=12)
ax.set_ylabel("Cantidad Vendida (semanal)", fontsize=12)
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig)

forecast_futuro = forecast[forecast['ds'] > fecha_corte].copy()
total_predicho = forecast_futuro['yhat'].sum() if not forecast_futuro.empty else 0
total_diario_estimado = total_predicho * (periodo_dias / (periodo_semanas * 7))

st.subheader(f"Total estimado para los próximos {periodo_dias} días:")
st.write(f"**{total_diario_estimado:.0f} unidades estimadas** para importar en {periodo_dias} días.")

df_eval = df_comparacion.dropna().copy()
df_eval = df_eval[df_eval['y'] > 0]

if df_eval.empty:
    st.warning("No hay suficientes datos reales > 0 para calcular métricas.")
else:
    y_true = df_eval['y']
    y_pred = df_eval['yhat']
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    st.write(f"**MAE semanal:** {mae:.2f}")
    st.write(f"**RMSE semanal:** {rmse:.2f}")
    st.write(f"**MAPE semanal:** {mape:.2f}%")
