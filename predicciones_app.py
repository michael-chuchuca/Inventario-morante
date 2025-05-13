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

# Slider para elegir días a predecir
periodo = st.slider("Selecciona el número de días a predecir:", min_value=7, max_value=90, value=45)

# Datos reales
real = df_item.set_index('FECHA_VENTA')['CANTIDAD_VENDIDA']
fecha_real_final = real.index[-1]
valor_real_final = real.values[-1]

# Predicción con Prophet
forecast = entrenar_prophet(df_item, periodo)
fecha_pred = forecast.index[-1]
valor_pred = forecast['yhat'].iloc[-1]

# Preparar datos para gráfico
df_p = df_item[['FECHA_VENTA', 'CANTIDAD_VENDIDA']].rename(columns={'FECHA_VENTA': 'ds', 'CANTIDAD_VENDIDA': 'y'})
forecast = forecast.reset_index()
forecast_futuro = forecast[forecast['ds'] > df_p['ds'].max()]

# -----------------------
# Gráfico
# -----------------------

fig, ax = plt.subplots(figsize=(10, 6))

# Estilo pastel y moderno
historico_color = '#A2C4C9'  # Azul pastel
prediccion_color = '#B6D7A8'  # Verde pastel
punto_real_color = '#3D85C6'  # Azul fuerte
punto_pred_color = '#38761D'  # Verde fuerte

# Curvas
ax.plot(df_p['ds'], df_p['y'], label='Histórico', color=historico_color, linewidth=2.5)
ax.plot(forecast_futuro['ds'], forecast_futuro['yhat'], label=f'Predicción ({periodo} días)', color=prediccion_color, linestyle='--', linewidth=2.5)

# Puntos finales
ax.plot(fecha_real_final, valor_real_final, 'o', color=punto_real_color, markersize=8, label='Último Real')
ax.plot(fecha_pred, valor_pred, 'o', color=punto_pred_color, markersize=8, label='Última Predicción')

# Anotaciones
ax.annotate(f'{valor_real_final:.0f}', (fecha_real_final, valor_real_final),
            textcoords="offset points", xytext=(0, 10), ha='center', fontsize=11, color=punto_real_color, fontweight='bold')
ax.annotate(f'{valor_pred:.0f}', (fecha_pred, valor_pred),
            textcoords="offset points", xytext=(0, 10), ha='center', fontsize=11, color=punto_pred_color, fontweight='bold')

# Línea de corte (inicio predicción)
fecha_corte = df_p['ds'].max()
ax.axvline(fecha_corte, color='gray', linestyle=':', alpha=0.6)
ax.annotate('Inicio de Predicción', xy=(fecha_corte, ax.get_ylim()[1]*0.9),
            xytext=(10, 0), textcoords='offset points', fontsize=10, color='gray')

# Estética
ax.set_title(f'Predicción de Demanda: {item_seleccionado}', fontsize=15, fontweight='bold')
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Cantidad Vendida', fontsize=12)
ax.grid(alpha=0.3)
ax.legend(frameon=False, fontsize=11)
plt.xticks(rotation=45)

st.pyplot(fig)

# -----------------------
# Total estimado
# -----------------------
st.subheader(f"Total estimado para los próximos {periodo} días:")
total_predicho = forecast_futuro['yhat'].sum()
st.write(f"🔢 **{total_predicho:.0f} unidades estimadas** para importar o producir en {periodo} días.")

# -----------------------
# Evaluación del último punto (opcional)
# -----------------------
st.subheader("Evaluación del Último Punto Predicho (referencial)")
mae = mean_absolute_error([valor_real_final], [valor_pred])
rmse = np.sqrt(mean_squared_error([valor_real_final], [valor_pred]))
mape = np.mean(np.abs((valor_real_final - valor_pred) / (valor_real_final + 1e-10))) * 100

st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAPE:** {mape:.2f}%")
