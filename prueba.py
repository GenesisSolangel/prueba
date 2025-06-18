import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client
import schedule
import threading
import time as tiempo
import uuid
import requests
from dotenv import load_dotenv
import os
import json
import folium
from streamlit_folium import st_folium
from folium.features import GeoJsonTooltip
import branca.colormap as cm
from branca.element import Template, MacroElement
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
import onnxruntime as ort


st.set_page_config(page_title="Red Eléctrica", layout="centered")

# Constantes de configuración de la API REE
BASE_URL = "https://apidatos.ree.es/es/datos/"

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
}

ENDPOINTS = {
    "demanda": ("demanda/evolucion", "hour"),
    "balance": ("balance/balance-electrico", "day"),
    "generacion": ("generacion/evolucion-renovable-no-renovable", "day"),
    "intercambios": ("intercambios/todas-fronteras-programados", "day"),
    "intercambios_baleares": ("intercambios/enlace-baleares", "day"),
}

# Cargar las variables de entorno desde el archivo .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ------------------------------ UTILIDADES ------------------------------

# Función para consultar un endpoint, según los parámetros dados, de la API de REE
def get_data(endpoint_name, endpoint_info, params):
    path, time_trunc = endpoint_info
    params["time_trunc"] = time_trunc
    url = BASE_URL + path

    try:
        response = requests.get(url, headers=HEADERS, params=params)
        # Si la búsqueda no fue bien, se devuelve una lista vacía
        if response.status_code != 200:
            return []
        response_data = response.json()
    except Exception:
        return []

    data = []

    # Verificamos si el item tiene "content" y asumimos que es una estructura compleja
    for item in response_data.get("included", []):
        attrs = item.get("attributes", {})
        category = attrs.get("title")

        if "content" in attrs:
            for sub in attrs["content"]:
                sub_attrs = sub.get("attributes", {})
                sub_cat = sub_attrs.get("title")
                for entry in sub_attrs.get("values", []):
                    entry["primary_category"] = category
                    entry["sub_category"] = sub_cat
                    data.append(entry)
        else:
            # Procesamos las estructuras más simples (demanda, generacion, intercambios_baleares), asumiendo que no hay subcategorías
            for entry in attrs.get("values", []):
                entry["primary_category"] = category
                entry["sub_category"] = None
                data.append(entry)

    return data


# Función para insertar cada DataFrame en Supabase
def insertar_en_supabase(nombre_tabla, df):
    df = df.copy()

    # Generamos IDs únicos
    df["record_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    # Convertimos fechas a string ISO
    for col in ["datetime", "extraction_timestamp"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Reemplazamos NaN por None
    # df = df.where(pd.notnull(df), None)

    # Convertir a lista de diccionarios e insertar
    data = df.to_dict(orient="records")

    try:
        supabase.table(nombre_tabla).insert(data).execute()
        print(f"✅ Insertados en '{nombre_tabla}': {len(data)} filas")
    except Exception as e:
        print(f"❌ Error al insertar en '{nombre_tabla}': {e}")


# ------------------------------ FUNCIONES DE DESCARGA ------------------------------
# Función de extracción de datos de los últimos x años, devuelve DataFrame. Ejecutar una vez al inicio para poblar la base de datos.
def get_data_for_last_x_years(num_years=3):
    all_dfs = []
    current_date = datetime.now()
    # Calculamos el año de inicio a partir del año actual
    start_year_limit = current_date.year - num_years

    # Iteramos sobre cada año y mes
    for year in range(start_year_limit, current_date.year + 1):
        for month in range(1, 13):
            # Si el mes es mayor al mes actual y el año es el actual, lo saltamos
            month_start = datetime(year, month, 1)
            if month_start > current_date:
                continue
            # Calculamos el final del mes, asegurándonos de no exceder la fecha actual
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(minutes=1)
            end_date_for_request = min(month_end, current_date)

            monthly_data = []  # para acumular todos los dfs del mes

            # Iteramos sobre cada endpoint y sacamos los datos
            for name, (path, granularity) in ENDPOINTS.items():
                params = {
                    "start_date": month_start.strftime("%Y-%m-%dT%H:%M"),
                    "end_date": end_date_for_request.strftime("%Y-%m-%dT%H:%M"),
                    "geo_trunc": "electric_system",
                    "geo_limit": "peninsular",
                    "geo_ids": "8741"
                }

                data = get_data(name, (path, granularity), params)

                if data:
                    df = pd.DataFrame(data)
                    # Lidiamos con problemas de zona horaria en la columna "datetime"
                    try:
                        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                    except Exception:
                        continue

                    # Obtenemos nuevas columnas y las reordenamos
                    df['year'] = df['datetime'].dt.year
                    df['month'] = df['datetime'].dt.month
                    df['day'] = df['datetime'].dt.day
                    df['hour'] = df['datetime'].dt.hour
                    df['extraction_timestamp'] = datetime.utcnow()
                    df['endpoint'] = name
                    df['record_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
                    df = df[['record_id', 'value', 'percentage', 'datetime',
                             'primary_category', 'sub_category', 'year', 'month',
                             'day', 'hour', 'endpoint', 'extraction_timestamp']]

                    monthly_data.append(df)
                    tiempo.sleep(1)

            # Generamos los dataframes individuales
            if monthly_data:
                df_nuevo = pd.concat(monthly_data, ignore_index=True)
                all_dfs.append(df_nuevo)

                tablas_dfs = {
                    "demanda": df_nuevo[df_nuevo["endpoint"] == "demanda"].drop(columns=["endpoint", "sub_category"],
                                                                                errors='ignore'),
                    "balance": df_nuevo[df_nuevo["endpoint"] == "balance"].drop(columns=["endpoint"], errors='ignore'),
                    "generacion": df_nuevo[df_nuevo["endpoint"] == "generacion"].drop(
                        columns=["endpoint", "sub_category"], errors='ignore'),
                    "intercambios": df_nuevo[df_nuevo["endpoint"] == "intercambios"].drop(columns=["endpoint"],
                                                                                          errors='ignore'),
                    "intercambios_baleares": df_nuevo[df_nuevo["endpoint"] == "intercambios_baleares"].drop(
                        columns=["endpoint", "sub_category"], errors='ignore'),
                }

                for tabla, df_tabla in tablas_dfs.items():
                    if not df_tabla.empty:
                        insertar_en_supabase(tabla, df_tabla)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# Función para actualizar los datos desde la API cada 24 horas
def actualizar_datos_desde_api():
    print(f"[{datetime.now()}] ⏳ Ejecutando extracción desde API...")
    current_date = datetime.now()
    start_date = current_date - timedelta(days=1)

    all_dfs = []

    for name, (path, granularity) in ENDPOINTS.items():
        params = {
            "start_date": start_date.strftime("%Y-%m-%dT%H:%M"),
            "end_date": current_date.strftime("%Y-%m-%dT%H:%M"),
            "geo_trunc": "electric_system",
            "geo_limit": "peninsular",
            "geo_ids": "8741"
        }

        datos = get_data(name, (path, granularity), params)

        if datos:
            df = pd.DataFrame(datos)
            try:
                df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            except Exception:
                continue

            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['hour'] = df['datetime'].dt.hour
            df['extraction_timestamp'] = datetime.utcnow()
            df['endpoint'] = name
            df['record_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

            df = df[['record_id', 'value', 'percentage', 'datetime',
                     'primary_category', 'sub_category', 'year', 'month',
                     'day', 'hour', 'endpoint', 'extraction_timestamp']]

            all_dfs.append(df)
            tiempo.sleep(1)
        else:
            print(f"⚠️ No se obtuvieron datos de '{name}'")

    if all_dfs:
        df_nuevo = pd.concat(all_dfs, ignore_index=True)

        tablas_dfs = {
            "demanda": df_nuevo[df_nuevo["endpoint"] == "demanda"].drop(columns=["endpoint", "sub_category"]),
            "balance": df_nuevo[df_nuevo["endpoint"] == "balance"].drop(columns=["endpoint"]),
            "generacion": df_nuevo[df_nuevo["endpoint"] == "generacion"].drop(columns=["endpoint", "sub_category"]),
            "intercambios": df_nuevo[df_nuevo["endpoint"] == "intercambios"].drop(columns=["endpoint"]),
            "intercambios_baleares": df_nuevo[df_nuevo["endpoint"] == "intercambios_baleares"].drop(
                columns=["endpoint", "sub_category"]),
        }

        for tabla, df in tablas_dfs.items():
            if not df.empty:
                insertar_en_supabase(tabla, df)


# Programador para actualizar datos desde la API cada 24 horas
def iniciar_programador_api():
    schedule.every(24).hours.do(actualizar_datos_desde_api)
    while True:
        schedule.run_pending()
        tiempo.sleep(60)


# threading.Thread(target=iniciar_programador_api, daemon=True).start()

# ------------------------------ CONSULTA SUPABASE ------------------------------

def get_data_from_supabase(table_name, start_date, end_date, page_size=1000):
    end_date += timedelta(days=1)
    start_iso = start_date.isoformat()
    end_iso = end_date.isoformat()

    all_data = []
    offset = 0
    while True:
        response = (
            supabase.table(table_name)
            .select("*")
            .gte("datetime", start_iso)
            .lte("datetime", end_iso)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        data = response.data
        if not data:
            break
        all_data.extend(data)
        offset += page_size
        if len(data) < page_size:
            break

    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

# ------------------------------ INTERFAZ ------------------------------

def main():
    st.title("Análisis de la Red Eléctrica Española")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Descripción", "Consulta de datos", "Visualización", "Predicciones", "Extras"])

    with tab2:  # Mueve el contexto de la tab2 aquí para que `modo` se defina antes de usarse en session_state
        st.subheader("Consulta de datos")

        modo = st.radio("Tipo de consulta:", ["Últimos días", "Año específico", "Histórico"], horizontal=True,
                        key="query_mode_radio")
        st.session_state["modo_seleccionado"] = modo  # Guardar el modo en session_state

        tabla = st.selectbox("Selecciona la tabla:", list(ENDPOINTS.keys()), key="query_table_select")

        df = pd.DataFrame()  # Inicializamos el DataFrame para evitar errores

        if modo == "Últimos días":
            dias = st.selectbox("¿Cuántos días atrás?", [7, 14, 30], key="query_days_select")
            end_date_query = datetime.now(timezone.utc)
            start_date_query = end_date_query - timedelta(days=dias)
            st.session_state["selected_year_for_viz"] = None  # Reset year if not in "Año específico" mode
        elif modo == "Año específico":
            current_year = datetime.now().year
            # Se usa `key` para que el selectbox mantenga su estado
            año = st.selectbox("Selecciona el año:", [current_year - i for i in range(3)], index=0,
                               key="query_year_select")
            st.session_state["selected_year_for_viz"] = año  # Store the selected year
            start_date_query = datetime(año, 1, 1, tzinfo=timezone.utc)
            end_date_query = datetime(año, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
        elif modo == "Histórico":
            # Si quieres cargar datos históricos de muchos años, ten cuidado con el rendimiento
            start_date_query = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # Ejemplo de fecha inicial
            end_date_query = datetime.now(timezone.utc)
            st.session_state["selected_year_for_viz"] = None  # Reset year if not in "Año específico" mode

        with st.spinner("Consultando Supabase..."):
            st.session_state["tabla_seleccionada_en_tab2"] = tabla
            df = get_data_from_supabase(tabla, start_date_query, end_date_query)

        # Mostrar resultados después de la consulta de cualquier modo
        if not df.empty:
            st.session_state["ree_data"] = df
            st.session_state[
                "tabla"] = tabla  # Esto es redundante si usas tabla_seleccionada_en_tab2, pero lo mantengo por seguridad
            st.write(f"Datos recuperados: {len(df)} filas")
            st.write("Último dato:", df['datetime'].max())
            st.success("Datos cargados correctamente desde Supabase.")
        else:
            st.warning("No se encontraron datos para ese período.")

    with tab1:  # Reordeno esto para que la tab2 se cargue primero y defina el estado
        st.subheader("¿Qué es esta app?")
        st.markdown("""
        Este proyecto explora los datos públicos de la **Red Eléctrica de España (REE)** a través de su API.
        Se analizan aspectos como:

        - La **demanda eléctrica** por hora.
        - El **balance eléctrico** por día.
        - La **generación** por mes.
        - Los **intercambios programados** con otros países.

        Estos datos permiten visualizar la evolución energética de España y generar análisis útiles para planificación y sostenibilidad.
        """)

    with tab3:
        st.subheader("Visualización")
        if "ree_data" in st.session_state and not st.session_state["ree_data"].empty:
            # Recuperamos el DataFrame principal de la sesión para el primer gráfico
            df = st.session_state["ree_data"]
            tabla = st.session_state["tabla_seleccionada_en_tab2"]  # Usamos la tabla seleccionada en tab2
            modo_actual = st.session_state.get("modo_seleccionado", "Últimos días")  # Obtener el modo

            if tabla == "demanda":
                fig = px.area(df, x="datetime", y="value", title="Demanda Eléctrica", labels={"value": "MW"})
                st.plotly_chart(fig, use_container_width=True)

                # --- Nuevo gráfico: Histograma de demanda con outliers para año específico ---
                if modo_actual == "Año específico":
                    año_seleccionado = st.session_state.get("selected_year_for_viz")
                    if año_seleccionado is None:
                        st.warning(
                            "Por favor, selecciona un año en la pestaña 'Consulta de datos' para ver el histograma de demanda.")
                    else:
                        st.subheader(f"Distribución de Demanda y Valores Atípicos para el año {año_seleccionado}")

                        # Filtra el DataFrame para el año seleccionado (df ya debe estar filtrado por el año, pero esto es por seguridad)
                        df_año = df[df['year'] == año_seleccionado].copy()

                        if not df_año.empty:
                            # Calcular Q1, Q3 y el IQR para la columna 'value' (demanda)
                            Q1 = df_año['value'].quantile(0.25)
                            Q3 = df_año['value'].quantile(0.75)
                            IQR = Q3 - Q1

                            # Calcular los límites de Tukey's Fence
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR

                            # Identificar valores atípicos
                            df_año['is_outlier'] = 'Normal'
                            df_año.loc[df_año['value'] < lower_bound, 'is_outlier'] = 'Atípico (bajo)'
                            df_año.loc[df_año['value'] > upper_bound, 'is_outlier'] = 'Atípico (alto)'

                            # Crear el histograma
                            fig_hist_outliers = px.histogram(
                                df_año,
                                x="value",
                                color="is_outlier",
                                title=f"Distribución Horaria de Demanda para {año_seleccionado}",
                                labels={"value": "Demanda (MW)", "is_outlier": "Tipo de Valor"},
                                category_orders={"is_outlier": ["Atípico (bajo)", "Normal", "Atípico (alto)"]},
                                color_discrete_map={'Normal': 'skyblue', 'Atípico (bajo)': 'orange',
                                                    'Atípico (alto)': 'red'},
                                nbins=50  # Ajusta el número de bins según la granularidad deseada
                            )
                            fig_hist_outliers.update_layout(bargap=0.1)  # Espacio entre barras
                            st.plotly_chart(fig_hist_outliers, use_container_width=True)

                            # Mostrar información sobre outliers
                            num_outliers_low = (df_año['is_outlier'] == 'Atípico (bajo)').sum()
                            num_outliers_high = (df_año['is_outlier'] == 'Atípico (alto)').sum()

                            if num_outliers_low > 0 or num_outliers_high > 0:
                                st.warning(
                                    f"Se han identificado {num_outliers_low} valores atípicos por debajo y {num_outliers_high} por encima (método IQR).")
                                st.info(f"Rango normal de demanda (IQR): {lower_bound:.2f} MW - {upper_bound:.2f} MW")
                            else:
                                st.info(
                                    "No se han identificado valores atípicos de demanda significativos (método IQR).")
                        else:
                            st.warning(
                                f"No hay datos de demanda para el año {año_seleccionado} para generar el histograma.")

                # --- Condición para mostrar la comparativa de años (mantenida de antes) ---
                if modo_actual == "Histórico":
                    st.subheader("Comparativa de Demanda entre años")

                    # Definir los dos años específicos para la comparación: el año pasado y el anterior
                    current_year = datetime.now().year

                    # Los años que queremos comparar: el año anterior al actual y el año anterior a ese
                    target_years_for_comparison = [current_year - 2, current_year - 1]

                    # Obtener todos los años disponibles en el DataFrame del modo histórico
                    all_available_years_in_df = sorted(list(df['year'].unique()))

                    # Filtrar solo los años que queremos comparar y que realmente están disponibles en el df
                    years_for_comparison = [
                        year for year in target_years_for_comparison
                        if year in all_available_years_in_df
                    ]

                    if len(years_for_comparison) == 2:
                        # Asegurarse de que están en el orden deseado (ej. [2023, 2024])
                        years_for_comparison.sort()
                    elif len(years_for_comparison) == 1:
                        st.info(
                            f"Solo se encontró un año de los deseados ({years_for_comparison[0]}) en modo histórico para la comparación. Se necesitan ambos años ({target_years_for_comparison[0]} y {target_years_for_comparison[1]}).")
                        years_for_comparison = []  # Vaciar para no intentar graficar
                    else:  # len(years_for_comparison) == 0
                        st.info(
                            f"No se encontraron datos para los años {target_years_for_comparison[0]} y {target_years_for_comparison[1]} en modo histórico para la comparación.")
                        years_for_comparison = []  # Vaciar para no intentar graficar

                    if years_for_comparison:  # Solo procede si tenemos al menos dos años para comparar
                        df_comparison_demanda = df.copy()  # Usamos el df ya cargado en modo histórico

                        # Nos aseguramos de que solo tengamos los años que queremos comparar
                        df_filtered_comparison = df_comparison_demanda[
                            df_comparison_demanda['year'].isin(years_for_comparison)].copy()

                        # Convertimos la columna 'datetime' a una fecha sin el año, para comparar día a día
                        # Esta 'sort_key' se usa para el gráfico horario
                        df_filtered_comparison['sort_key'] = df_filtered_comparison['datetime'].apply(
                            lambda dt: dt.replace(year=2000)  # Usar un año base para ordenar correctamente
                        )
                        df_filtered_comparison = df_filtered_comparison.sort_values('sort_key')

                        # --- Gráfico de Demanda Horaria General Comparativa ---
                        fig_comp_hourly = px.line(
                            df_filtered_comparison,
                            x="sort_key",  # Usamos la 'sort_key' que es datetime
                            y="value",
                            color="year",
                            title="Demanda Horaria - Comparativa",
                            labels={"sort_key": "Mes y Día", "value": "Demanda (MW)", "year": "Año"},
                            hover_data={"year": True, "datetime": "|%Y-%m-%d %H:%M"}
                        )
                        fig_comp_hourly.update_xaxes(tickformat="%b %d")  # Formato para mostrar Mes y Día en el eje X
                        st.plotly_chart(fig_comp_hourly, use_container_width=True)

                        # --- Gráficos de Comparación de Métricas Diarias (Media, Mediana, Mínima, Máxima) ---
                        # Agrupar por 'year' y 'month-day' para obtener las métricas diarias para cada año
                        metrics_comp = df_filtered_comparison.groupby(
                            ['year', df_filtered_comparison['datetime'].dt.strftime('%m-%d')])['value'].agg(
                            ['mean', 'median', 'min', 'max']).reset_index()
                        metrics_comp.columns = ['year', 'month_day', 'mean', 'median', 'min', 'max']

                        # La corrección para el ValueError: day is out of range for month está aquí
                        metrics_comp['sort_key'] = pd.to_datetime('2000-' + metrics_comp['month_day'],
                                                                  format='%Y-%m-%d')
                        metrics_comp = metrics_comp.sort_values('sort_key')

                        metric_names = {
                            'mean': 'Media diaria de demanda',
                            'median': 'Mediana diaria de demanda',
                            'min': 'Mínima diaria de demanda',
                            'max': 'Máxima diaria de demanda',
                        }

                        for metric in ['mean', 'median', 'min', 'max']:
                            fig = px.line(
                                metrics_comp,
                                x="sort_key",  # <--- CAMBIO CLAVE: Usar 'sort_key' (tipo datetime) para el eje X
                                y=metric,
                                color="year",
                                title=metric_names[metric],
                                labels={"sort_key": "Fecha (Mes-Día)", metric: "Demanda (MW)", "year": "Año"},
                                # <--- CAMBIO EN ETIQUETA
                            )
                            fig.update_xaxes(tickformat="%b %d")  # Formato para mostrar solo Mes y Día
                            # Si las líneas siguen entrecortadas, considera añadir `connectgaps=True`
                            # fig.update_traces(connectgaps=True)
                            st.plotly_chart(fig, use_container_width=True)


                    else:
                        st.warning(f"No hay suficientes datos de Demanda disponibles para la comparación.")

                    # --- Gráfico de Identificación de años outliers (mantenida de antes) ---
                    st.subheader("Identificación de Años Outliers (Demanda Anual Total)")

                    st.markdown(
                        "**Este gráfico muestra los años identificados como outliers en la demanda total anual.**\n\n"
                        "En este caso, solo se detecta como outlier el año **2025**, lo cual es esperable ya que todavía no ha finalizado "
                        "y su demanda acumulada es significativamente menor.\n\n"
                        "Los años **2022, 2023 y 2024** presentan una demanda anual muy similar, en torno a los **700 MW**, por lo que "
                        "no se consideran outliers según el criterio del rango intercuartílico (IQR)."
                    )

                    # Asegurarse de que el df tiene la columna 'year'
                    if 'year' not in df.columns:
                        df['year'] = df['datetime'].dt.year

                    # Agrupar por año para obtener la demanda total anual
                    df_annual_summary = df.groupby('year')['value'].sum().reset_index()
                    df_annual_summary.rename(columns={'value': 'total_demand_MW'}, inplace=True)

                    if not df_annual_summary.empty and len(df_annual_summary) > 1:
                        # Calcular Q1, Q3 y el IQR
                        Q1 = df_annual_summary['total_demand_MW'].quantile(0.25)
                        Q3 = df_annual_summary['total_demand_MW'].quantile(0.75)
                        IQR = Q3 - Q1

                        # Calcular los límites para los outliers
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        # Identificar los años que son outliers
                        df_annual_summary['is_outlier'] = (
                                (df_annual_summary['total_demand_MW'] < lower_bound) |
                                (df_annual_summary['total_demand_MW'] > upper_bound)
                        )

                        # Crear el gráfico de barras
                        fig_outliers = px.bar(
                            df_annual_summary,
                            x='year',
                            y='total_demand_MW',
                            color='is_outlier',  # Colorear las barras si son outliers
                            title='Demanda Total Anual y Años Outlier',
                            labels={'total_demand_MW': 'Demanda Total Anual (MW)', 'year': 'Año',
                                    'is_outlier': 'Es Outlier'},
                            color_discrete_map={False: 'skyblue', True: 'red'}  # Definir colores
                        )

                        st.plotly_chart(fig_outliers, use_container_width=True)

                        # Mostrar los años identificados como outliers
                        outlier_years = df_annual_summary[df_annual_summary['is_outlier']]['year'].tolist()
                        if outlier_years:
                            st.warning(
                                f"Se han identificado los siguientes años como outliers: {', '.join(map(str, outlier_years))}")
                        else:
                            st.info("No se han identificado años outliers significativos (según el método IQR).")
                    elif not df_annual_summary.empty and len(df_annual_summary) <= 1:
                        st.info("Se necesitan al menos 2 años de datos para calcular outliers de demanda anual.")
                    else:
                        st.warning("No hay datos anuales disponibles para calcular outliers.")
                # El siguiente 'else' se aplica si modo_actual NO es "Histórico"
                elif modo_actual != "Histórico" and modo_actual != "Año específico":
                    st.info(
                        "Selecciona el modo 'Histórico' para ver la comparativa de años y la identificación de outliers anuales, o 'Año específico' para el histograma de demanda con outliers.")

            elif tabla == "balance":

                df_balance = df.groupby([df["datetime"].dt.date, "primary_category"])["value"].sum().reset_index()
                df_balance.rename(columns={"datetime": "date"}, inplace=True)
                df_balance = df_balance.sort_values("date")
                fig = px.line(
                    df_balance,
                    x="date",
                    y="value",
                    color="primary_category",
                    title="Balance Eléctrico Diario"
                )

                fig.update_layout(
                    xaxis_title="Fecha",
                    yaxis_title="MWh",
                    legend_title="Categoría",
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    "**Balance eléctrico diario por categoría**\n\n"
                    "Este gráfico representa el balance energético entre las distintas fuentes y usos diarios. Cada barra agrupa los componentes "
                    "principales del sistema: generación, consumo, pérdidas y exportaciones.\n\n"
                    "Es útil para entender si hay superávit, déficit o equilibrio en la red cada día, y cómo se distribuye el uso de energía entre sectores."
                )

            elif tabla == "generacion":
                df['date'] = df['datetime'].dt.date  # Para reducir a nivel diario (si no lo tienes)

                df_grouped = df.groupby(['date', 'primary_category'])['value'].sum().reset_index()

                fig = px.line(
                    df_grouped,
                    x="date",
                    y="value",
                    color="primary_category",
                    title="Generación diaria agregada por tipo"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    "**Generación diaria agregada por tipo**\n\n"
                    "Se visualiza la evolución de la generación eléctrica por fuente: renovables (eólica, solar, hidroeléctrica) y no renovables "
                    "(gas, nuclear, etc.).\n\n"
                    "Esta gráfica permite observar patrones como aumentos de producción renovable en días soleados o ventosos, así como la estabilidad "
                    "de tecnologías de base como la nuclear. Es clave para analizar la transición energética."
                )


            elif tabla == "intercambios":
                st.subheader("Mapa Coroplético de Intercambios Eléctricos")

                st.markdown(
                    "**Intercambios eléctricos internacionales**\n\n"
                    "Este mapa muestra el **saldo neto de energía** (exportaciones menos importaciones) entre España y los países vecinos: "
                    "**Francia, Portugal, Marruecos y Andorra**.\n\n"
                    "Los valores positivos indican que **España exporta más energía de la que importa**, mientras que los negativos reflejan lo contrario.\n\n"
                    "Este análisis es clave para comprender el papel de España como nodo energético regional, identificar dependencias o excedentes, "
                    "y analizar cómo varían los flujos en situaciones especiales como picos de demanda o apagones."
                )

                # Agrupar y renombrar columnas
                df_map = df.groupby("primary_category")["value"].sum().reset_index()
                df_map.columns = ["pais_original", "Total"]

                # Mapeo de nombres a inglés
                nombre_map = {
                    "francia": "France",
                    "portugal": "Portugal",
                    "andorra": "Andorra",
                    "marruecos": "Morocco"
                }
                df_map["Country"] = df_map["pais_original"].map(nombre_map)
                df_map = df_map.dropna(subset=["Country"])

                # Crear diccionario país → saldo
                country_data = df_map.set_index("Country")["Total"].to_dict()

                # Cargar GeoJSON
                with open("world_countries_with_andorra.json", "r", encoding="utf-8") as f:
                    world_geo = json.load(f)

                # Crear mapa base
                world_map = folium.Map(location=[40, 0], zoom_start=5)

                # Establecer rango fijo para que el color blanco siempre sea cero
                vmin = -8000000  # Ajusta según el rango máximo esperado de exportación
                vmax = 8000000  # Ajusta según el rango máximo esperado de importación

                colormap = cm.LinearColormap(
                    colors=["blue", "white", "red"],
                    vmin=vmin, vmax=vmax
                )

                # Insertar saldo como propiedad en el GeoJSON
                for feature in world_geo["features"]:
                    country_name = feature["properties"]["name"]
                    saldo = country_data.get(country_name)
                    feature["properties"]["saldo"] = saldo if saldo is not None else "No disponible"

                # Añadir capa GeoJson con estilos y tooltips
                folium.GeoJson(
                    world_geo,
                    style_function=lambda feature: {
                        'fillColor': colormap(feature["properties"]["saldo"])
                        if isinstance(feature["properties"]["saldo"], (int, float)) else 'black',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7 if isinstance(feature["properties"]["saldo"], (int, float)) else 0.3,
                    },
                    tooltip=GeoJsonTooltip(
                        fields=['name', 'saldo'],
                        aliases=['País:', 'Saldo (MWh):'],
                        labels=True,
                        sticky=True,
                        localize=True,
                        toLocaleString=True
                    ),
                    highlight_function=lambda x: {'weight': 3, 'color': 'blue'}
                ).add_to(world_map)

                # Crear leyenda personalizada como MacroElement
                legend_html = f"""
                    {{% macro html(this, kwargs) %}}

                    <div style="
                        position: fixed;
                        bottom: 50px;
                        left: 50px;
                        width: 250px;
                        height: 120px;
                        background-color: white;
                        border:2px solid grey;
                        z-index:9999;
                        font-size:14px;
                        padding: 10px;
                        ">
                        <b>Saldo neto de energía (MWh)</b><br><br>
                        <div style="
                            width: 200px;
                            height: 20px;
                            background: linear-gradient(to right, blue, white, red);
                            border: 1px solid black;
                        "></div>
                        <div style="display: flex; justify-content: space-between; width: 200px;">
                            <span style="font-size:12px;">{vmin} MWh</span>
                            <span style="font-size:12px;">0</span>
                            <span style="font-size:12px;">{vmax} MWh</span>
                        </div>
                    </div>

                    {{% endmacro %}}
                    """

                legend = MacroElement()
                legend._template = Template(legend_html)
                world_map.get_root().add_child(legend)

                st_folium(world_map, width=1285, height=600)

                st.markdown(
                    "**Mapa de intercambios internacionales de energía – Contexto del apagón del 28 de abril de 2025**\n\n"
                    "Este mapa revela cómo se comportaron los **flujos internacionales de energía** en torno al **apagón del 28 de abril de 2025**.\n\n"
                    "Una **disminución en los intercambios con Francia o Marruecos** podría indicar una disrupción en el suministro internacional "
                    "o un corte de emergencia.\n\n"
                    "Si **España aparece como exportadora neta incluso durante el apagón**, esto sugiere que el problema no fue de generación, "
                    "sino posiblemente **interno** (fallo en la red o desconexión de carga).\n\n"
                    "La inclusión de **Andorra y Marruecos** proporciona un contexto más completo del comportamiento eléctrico en la península "
                    "y el norte de África.\n\n"
                    "Este gráfico es crucial para analizar si los intercambios internacionales actuaron de forma inusual, lo cual puede dar pistas "
                    "sobre causas externas o coordinación regional durante el evento."
                )
            elif tabla == "intercambios_baleares":
                # Filtramos las dos categorías
                df_ib = df[df['primary_category'].isin(['Entradas', 'Salidas'])].copy()

                # Agregamos por fecha para evitar múltiples por hora si fuera el caso
                df_ib_grouped = df_ib.groupby(['datetime', 'primary_category'])['value'].sum().reset_index()

                df_ib_grouped['value'] = df_ib_grouped['value'].abs()
                st.markdown(
                    "**Intercambios de energía con Baleares (Primer semestre 2025)**\n\n"
                    "Durante el primer semestre de **2025**, las **salidas de energía hacia Baleares** superan consistentemente a las entradas, "
                    "lo que indica que el sistema peninsular actúa mayormente como **exportador neto de energía**.\n\n"
                    "Ambos flujos muestran una **tendencia creciente hacia junio**, especialmente las salidas, lo que podría reflejar un aumento "
                    "en la demanda en Baleares o una mejora en la capacidad exportadora del sistema."
                )

                fig = px.area(
                    df_ib_grouped,
                    x="datetime",
                    y="value",
                    color="primary_category",
                    labels={"value": "Energía (MWh)", "datetime": "Fecha"},
                    title="Intercambios con Baleares - Área Apilada (Magnitud)"
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.line(df, x="datetime", y="value", title="Visualización")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver datos en tabla"):
                st.dataframe(df, use_container_width=True)
        else:
            st.info("Consulta primero los datos desde la pestaña anterior.")

    with tab4:
        # -------------------------------------------
        # CONFIGURACIÓN DE STREAMLIT
        # -------------------------------------------
        st.title("Predicción de Series Temporales con Modelos Preentrenados")

        # -------------------------------------------
        # SELECCIÓN DE MODELO Y PÉRDIDA
        # -------------------------------------------
        model_type = st.selectbox("Selecciona el modelo", ["SimpleRNN", "LSTM", "GRU"])
        loss_function = st.selectbox("Función de pérdida", ["mse", "mae"])

        model_filename = f'models/{model_type}_model_{loss_function}.onnx'
        scaler_filename = f'models/scaler_{model_type}_{loss_function}.pkl'
        history_filename = f'models/{model_type}_history_{loss_function}.pkl'

        try:
            # Cargar modelo ONNX
            session  = ort.InferenceSession(model_filename)

            # Cargar scaler
            with open(scaler_filename, 'rb') as f:
                scaler = pickle.load(f)

            # Cargar history
            with open(history_filename, 'rb') as f:
                history = pickle.load(f)

            st.success(f"Modelo {model_filename} cargado correctamente.")

            # -------------------------------------------
            # GRÁFICO DE FUNCIÓN DE PÉRDIDA
            # -------------------------------------------
            st.subheader("Gráfico de la función de pérdida")
            df_loss = pd.DataFrame({
                'epoch': range(1, len(history['loss']) + 1),
                'train_loss': history['loss'],
                'val_loss': history['val_loss']
            })
            fig_loss = px.line(df_loss, x='epoch', y=['train_loss', 'val_loss'],
                               labels={'value': 'Loss', 'epoch': 'Época'},
                               title='Evolución de la pérdida durante el entrenamiento')
            st.plotly_chart(fig_loss, use_container_width=True)

            # -------------------------------------------
            # CARGA DE LOS DATOS
            # -------------------------------------------
            df_prediccion = pd.read_csv('datos_prediccion.csv')
            df_prediccion['datetime'] = pd.to_datetime(df_prediccion['datetime'])
            df_prediccion = df_prediccion.set_index('datetime')

            df_prediccion['value_scaled'] = scaler.transform(df_prediccion[['value']])

            # Preparación de datos
            n_pasos = 24

            def crear_secuencias(datos, n_pasos):
                X, y = [], []
                for i in range(len(datos) - n_pasos):
                    X.append(datos[i:i + n_pasos])
                    y.append(datos[i + n_pasos])
                return np.array(X), np.array(y)

            X, y = crear_secuencias(df_prediccion['value_scaled'].values, n_pasos)
            X = X.reshape((X.shape[0], X.shape[1], 1)).astype('float32')

            # Preparar la sesión ONNX
            input_name = session.get_inputs()[0].name

            # -------------------------------------------
            # ONE-STEP PREDICTION
            # -------------------------------------------
            st.subheader("One-Step Prediction")

            y_pred_scaled = []
            for i in range(X.shape[0]):
                pred = session.run(None, {input_name: X[i:i + 1]})[0]
                y_pred_scaled.append(pred[0][0])

            y_pred_scaled = np.array(y_pred_scaled).reshape(-1, 1)
            y_pred = scaler.inverse_transform(y_pred_scaled)
            y_real = scaler.inverse_transform(y.reshape(-1, 1))

            df_pred = pd.DataFrame({
                'Real': y_real.flatten(),
                'Predicción': y_pred.flatten()
            }, index=df_prediccion.index[n_pasos:])

            fig_pred = px.line(df_pred.head(200), title="Predicción vs Real (One-Step)")
            st.plotly_chart(fig_pred, use_container_width=True)

            mse = mean_squared_error(y_real, y_pred)
            st.metric(label="MSE (Error cuadrático medio)", value=f"{mse:.2f}")

            # -------------------------------------------
            # MULTI-STEP PREDICTION
            # -------------------------------------------
            st.subheader("Multi-Step Prediction")

            # Elegir número de pasos a predecir
            n_pred = st.slider("Número de pasos a predecir (Multi-Step)", min_value=1, max_value=168, value=24, step=1)

            ultimos_valores = df_prediccion['value_scaled'].values[-n_pasos:].tolist()
            predicciones_multi = []

            for _ in range(n_pred):
                entrada = np.array(ultimos_valores[-n_pasos:]).reshape((1, n_pasos, 1)).astype('float32')
                pred_scaled = session.run(None, {input_name: entrada})[0][0][0]
                predicciones_multi.append(pred_scaled)
                ultimos_valores.append(pred_scaled)

            predicciones_multi = scaler.inverse_transform(np.array(predicciones_multi).reshape(-1, 1)).flatten()

            fechas_futuras = pd.date_range(start=df_prediccion.index[-1] + pd.Timedelta(hours=1), periods=n_pred,
                                           freq='H')

            fig_multi = go.Figure()
            fig_multi.add_trace(
                go.Scatter(x=df_prediccion.index, y=df_prediccion['value'], mode='lines', name='Datos reales'))
            fig_multi.add_trace(
                go.Scatter(x=fechas_futuras, y=predicciones_multi, mode='lines+markers', name='Predicción Multi-Step'))

            fig_multi.update_layout(title="Predicción Multi-Step", xaxis_title="Fecha", yaxis_title="Valor")
            st.plotly_chart(fig_multi, use_container_width=True)

        except Exception as e:
            st.warning(f"❌ El modelo {model_type} con pérdida {loss_function} no se encuentra o ocurrió un error.\n{e}")

    with tab5:
        if tabla == "demanda":

            # --- HEATMAP ---
            df_heatmap = df.copy()
            df_heatmap['weekday'] = df_heatmap['datetime'].dt.day_name()
            df_heatmap['hour'] = df_heatmap['datetime'].dt.hour

            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            heatmap_data = (
                df_heatmap.groupby(['weekday', 'hour'])['value']
                .mean()
                .reset_index()
                .pivot(index='weekday', columns='hour', values='value')
                .reindex(days_order)
            )
            st.markdown(
                "**Demanda promedio por día y hora**\n\n"
                "La demanda eléctrica promedio es más alta entre semana, especialmente de **lunes a viernes**, "
                "con picos concentrados entre las **7:00 y 21:00 horas**. El máximo se registra los **viernes alrededor de las 19:00 h**, "
                "superando los **32 000 MW**.\n\n"
                "En contraste, los **fines de semana** muestran una demanda notablemente más baja y estable."
            )
            fig1 = px.imshow(
                heatmap_data,
                labels=dict(x="Hora del día", y="Día de la semana", color="Demanda promedio (MW)"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale="YlGnBu",
                aspect="auto",
            )
            fig1.update_layout(title="Demanda promedio por día y hora")

            st.plotly_chart(fig1, use_container_width=True)

            # --- BOXPLOT ---
            df_box = df.copy()

            df_box["month"] = df_box["datetime"].dt.month
            st.markdown(
                "**Distribución de Demanda por mes (2025)**\n\n"
                "La demanda eléctrica presenta **mayor variabilidad y valores más altos en los primeros tres meses del año**, "
                "especialmente en **enero**.\n\n"
                "En **abril**, se observa una mayor cantidad de valores atípicos a la baja, lo cual coincide con el "
                "**apagón nacional del 28/04/2025**, donde España estuvo sin luz durante aproximadamente 8 a 10 horas.\n\n"
                "A partir de **mayo**, la demanda se estabiliza ligeramente, con una reducción progresiva en la mediana mensual."
            )
            fig2 = px.box(
                df_box,
                x="month",
                y="value",
                title="Distribución de Demanda por mes",
                labels={"value": "Demanda (MWh)", "hour": "Hora del Día"}
            )

            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.markdown("Nada que ver... de momento")


if __name__ == "__main__":
    main()
