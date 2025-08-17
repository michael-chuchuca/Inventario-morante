# ===============================
# Importación de librerías
# ===============================
import streamlit as st  # Librería para crear aplicaciones web interactivas
import pandas as pd     # Para manipulación de datos en formato tabla
import numpy as np      # Para cálculos numéricos y estadísticos
from prophet import Prophet  # Modelo de predicción de series temporales
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Métricas de evaluación
import matplotlib.pyplot as plt  # Para generar gráficos

# ===============================
# Definición de funciones
# ===============================

# Función para cargar los datos desde un archivo Excel
@st.cache_data  # Cachea los datos para mejorar el rendimiento
def cargar_datos(excel_path):
    df = pd.read_excel(excel_path)  # Carga el archivo Excel
    df.columns = df.columns.str.strip()  # Elimina espacios en los nombres de columnas
    df['FECHA_VENTA'] = pd.to_datetime(df['FECHA_VENTA'])  # Convierte la columna de fechas
    df = df.sort_values(by='FECHA_VENTA')  # Ordena los datos cronológicamente
    return df

# Función para preparar la serie temporal semanal de un ítem
def preparar_serie_semanal(df_item_raw):
    # Agrupa por semana y suma las cantidades vendidas
    df_agg = df_item_raw.groupby(pd.Grouper(key='FECHA_VENTA', freq='W')).agg({
        'CANTIDAD_VENDIDA': 'sum',
        'DESCRIPCION': 'first',
        'ITEM': 'first'
    }).reset_index()

    # Renombra columnas para que Prophet las reconozca
    df_agg = df_agg.rename(columns={'FECHA_VENTA': 'ds', 'CANTIDAD_VENDIDA': 'y'})

    # Relleno de semanas faltantes con 0 ventas
    fecha_inicio = df_agg['ds'].min()
    fecha_fin = df_agg['ds'].max()
    todas_las_fechas = pd.date_range(fecha_inicio, fecha_fin, freq='W')
    df_agg = df_agg.set_index('ds').reindex(todas_las_fechas).fillna({'y': 0}).reset_index()
    df_agg = df_agg.rename(columns={'index': 'ds'})

    # Elimina valores extremos (outliers) del 2% superior
    umbral_extremo = df_agg['y'].quantile(0.98)
    df_agg['y'] = np.where(df_agg['y'] > umbral_extremo, umbral_extremo, df_agg['y'])

    # Suavizado doble para reducir ruido en la serie
    df_agg['y'] = df_agg['y'].rolling(window=3, min_periods=1).mean()
    df_agg['y'] = df_agg['y'].clip(upper=df_agg['y'].quantile(0.95))
    df_agg['y'] = df_agg['y'].rolling(window=2, min_periods=1).mean()

    # Recupera columnas perdidas tras el reindex
    df_agg['DESCRIPCION'] = df_item_raw['DESCRIPCION'].iloc[0]
    df_agg['ITEM'] = df_item_raw['ITEM'].iloc[0]

    return df_agg

# Función para entrenar el modelo Prophet y generar predicciones
def entrenar_prophet_semanal(df, periodo_semanas):
    model = Prophet(
        weekly_seasonality=True,  # Activa estacionalidad semanal
        yearly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',  # Modelo aditivo
        changepoint_prior_scale=0.01,  # Sensibilidad a cambios estructurales
        seasonality_prior_scale=1.0,
        changepoint_range=0.9  # Porcentaje del historial usado para detectar cambios
    )
    model.fit(df)  # Entrena el modelo con los datos históricos
    future = model.make_future_dataframe(periods=periodo_semanas, freq='W')  # Genera fechas futuras
    forecast = model.predict(future)  # Realiza la predicción
    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))  # Evita valores negativos
    return forecast[['ds', 'yhat']]

# ===============================
# Interfaz de usuario con Streamlit
# ===============================

# Título principal de la aplicación
st.markdown("<h1 style='text-align: center;'>Predicción de Demanda Semanal con Prophet</h1>", unsafe_allow_html=True)

# Carga de datos desde archivo Excel
excel_path = "Items_Morante.xlsx"
df = cargar_datos(excel_path)

# Combina código de ítem y descripción para mostrar en el selector
df['ITEM_DESC'] = df['ITEM'].astype(str) + " - " + df['DESCRIPCION'].astype(str)
item_opciones = df[['ITEM', 'ITEM_DESC']].drop_duplicates().set_index('ITEM_DESC')

# Selector de ítem en la interfaz
item_seleccionado_desc = st.selectbox("Selecciona un ítem:", item_opciones.index)
item_seleccionado = item_opciones.loc[item_seleccionado_desc]['ITEM']

# Filtra los datos del ítem seleccionado
df_item_raw = df[df['ITEM'] == item_seleccionado].copy()
descripcion = df_item_raw['DESCRIPCION'].iloc[0]

# Selector de periodo de predicción en días
periodo_dias = st.slider("Selecciona el número de días a predecir:", min_value=7, max_value=90, value=45)
periodo_semanas = int(np.ceil(periodo_dias / 7))  # Convierte días a semanas

# Prepara la serie temporal y realiza la predicción
df_semanal = preparar_serie_semanal(df_item_raw)
forecast = entrenar_prophet_semanal(df_semanal, periodo_semanas)

# Junta datos reales y predicción para comparar
df_real = df_semanal.copy()
df_comparacion = pd.merge(df_real, forecast, on='ds', how='left')
fecha_corte = df_real['ds'].max()  # Última fecha real

# ===============================
# Visualización del pronóstico
# ===============================

fig, ax = plt.subplots(figsize=(14, 6))

# Serie original sin suavizado
ax.plot(df_comparacion['ds'], df_item_raw.groupby(pd.Grouper(key='FECHA_VENTA', freq='W'))
        ['CANTIDAD_VENDIDA'].sum().reset_index()['CANTIDAD_VENDIDA'], 'c--', label='Serie Original', linewidth=1.2)

# Serie suavizada real
ax.plot(df_comparacion['ds'], df_comparacion['y'], 'b-', label='Cantidad Real', linewidth=2)

# Serie pronosticada
ax.plot(forecast['ds'], forecast['yhat'], 'r--', label='Cantidad Pronosticada', linewidth=2)

# Línea vertical que indica el inicio de la predicción
ax.axvline(fecha_corte, color='gray', linestyle=':', alpha=0.7)
ax.annotate('Inicio de Predicción', xy=(fecha_corte, ax.get_ylim()[1]*0.9),
            xytext=(10, 0), textcoords='offset points', fontsize=10, color='gray')

# Configuración del gráfico
ax.set_title("Pronóstico Semanal de Ventas con Prophet Optimizado", fontsize=15)
ax.set_xlabel("Fecha", fontsize=12)
ax.set_ylabel("Cantidad Vendida (semanal)", fontsize=12)
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig)

# ===============================
# Estimación total futura
# ===============================

# Filtra las fechas futuras del pronóstico
forecast_futuro = forecast[forecast['ds'] > fecha_corte].copy()
total_predicho = forecast_futuro['yhat'].sum() if not forecast_futuro.empty else 0

# Estima el total diario proporcional
total_diario_estimado = total_predicho * (periodo_dias / (periodo_semanas * 7))

# Muestra el total estimado en la interfaz
st.subheader(f"Total estimado para los próximos {periodo_dias} días:")
st.write(f"**{total_diario_estimado:.0f} unidades estimadas** para importar en {periodo_dias} días.")

# ===============================
# Evaluación del modelo
# ===============================

# Filtra datos válidos para evaluación
df_eval = df_comparacion.dropna().copy()
df_eval = df_eval[df_eval['y'] > 0]

# Verifica si hay suficientes datos para calcular métricas
if df_eval.empty:
    st.warning("No hay suficientes datos reales > 0 para calcular métricas.")
else:
    y_true = df_eval['y']  # Valores reales
    y_pred = df_eval['yhat']  # Valores pronosticados

    # Cálculo de métricas de error
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Muestra las métricas en la interfaz
    st.write(f"**MAE semanal:** {mae:.2f}")
    st.write(f"**RMSE semanal:** {rmse:.2f}")
    st.write(f"**MAPE semanal:** {mape:.2f}%")
