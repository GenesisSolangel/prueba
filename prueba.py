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
import joblib
from prophet.plot import plot_plotly, plot_components_plotly


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
    st.title("Análisis y Predicción de la Red Eléctrica Española (REE)")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Descripción","Visualización","Comparador","Predicciones: RNN","Predicciones: Prophet","Extras","Quiénes somos"])

    with st.sidebar:
        st.header("Filtros de consulta de datos")

        modo = st.radio("Tipo de consulta:", ["Últimos días", "Año específico", "Histórico"], horizontal=False,
                        key="query_mode_radio")
        st.session_state["modo_seleccionado"] = modo  # Guardar el modo en session_state

        tabla = st.selectbox("Selecciona la tabla:", list(ENDPOINTS.keys()), key="query_table_select")

        df = pd.DataFrame()  # Inicializamos el DataFrame para evitar errores

        if modo == "Últimos días":
            dias = st.selectbox("¿Cuántos días atrás?", [7, 14, 30], key="query_days_select")
            end_date_query = datetime.now(timezone.utc)
            start_date_query = end_date_query - timedelta(days=dias)
            st.session_state["selected_year_for_viz"] = None
        elif modo == "Año específico":
            current_year = datetime.now().year
            año = st.selectbox("Selecciona el año:", [current_year - i for i in range(3)], index=0,
                               key="query_year_select")
            st.session_state["selected_year_for_viz"] = año
            start_date_query = datetime(año, 1, 1, tzinfo=timezone.utc)
            end_date_query = datetime(año, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
        elif modo == "Histórico":
            start_date_query = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            end_date_query = datetime.now(timezone.utc)
            st.session_state["selected_year_for_viz"] = None

        with st.spinner("Consultando Supabase..."):
            st.session_state["tabla_seleccionada_en_tab2"] = tabla
            df = get_data_from_supabase(tabla, start_date_query, end_date_query)

        if not df.empty:
            st.session_state["ree_data"] = df
            st.session_state["tabla"] = tabla
            st.write(f"Datos recuperados: {len(df)} filas")
            st.write("Último dato:", df['datetime'].max())
            st.success("Datos cargados correctamente desde Supabase.")
        else:
            st.warning("No se encontraron datos para ese período.")

    with tab1:
        st.markdown("""
        ## ¿Qué es esta app?
        Esta aplicación interactiva permite consultar, visualizar y predecir datos públicos de la **Red Eléctrica de España (REE)**,  utilizando datos históricos proporcionados por su API. Se analizan aspectos como:

        - La **demanda eléctrica** por hora.
        - El **balance eléctrico** por día.
        - La **generación** por mes.
        - Los **intercambios programados** con otros países.
        
        ## Objetivos de la aplicación
        - Explorar la evolución de estos aspectos en diferentes periodos de tiempo.
        - Comparar métricas de demanda entre años específicos y detectar posibles años atípicos (outliers).
        - Realizar predicciones de los valores de demanda mediante diferentes modelos de deep learning.

        ## ¿Cómo funciona?
        - **Supabase:** Se usa como base de datos para almacenar y consultar la información histórica de forma eficiente.
        - **Kaggle:** Se emplea como entorno de entrenamiento para los modelos de predicción (RNN y Prophet).
        - **Streamlit:** Permite el desarrollo de esta app en la nube, ofreciendo interacción por parte del usuario.""")

        st.image("diagrama_app.png", caption="Flujo de la aplicación y conexión entre módulos", use_container_width  =True)
        st.image("supabase_schema.png", caption="Esquema de la base de datos en Supabase", use_container_width  =True)

        st.markdown("""
        ## Secciones de navegación:
        - **Descripción:** Página de inicio con la descripción general del proyecto.
        - **Filtros de consulta de datos:** Barra lateral para filtrar los datos según el interés del usuario.
        - **Visualización:** Análisis gráfico de los aspectos históricos comentados anteriormente.
        - **Comparador:** Comparación de la demanda entre dos años seleccionables y detección de outliers.
        - **Predicciones RNN:** Predicciones generadas mediante Redes Neuronales Recurrentes.
        - **Predicciones Prophet:** Predicciones usando el modelo Prophet de Facebook.
        - **Extras:** Análisis complementarios de interés.
        - **Quiénes somos:** Información sobre el equipo.
        """)

    with tab2:
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

    with tab3:
        st.subheader("Comparador de demanda eléctrica entre años")

        if modo_actual != "Histórico" or tabla != "demanda":
            st.info(
                "Para realizar la comparación, primero consulta datos de la tabla 'Demanda' en modo 'Histórico' en la barra lateral de consulta de datos.")
        elif "ree_data" in st.session_state and not st.session_state["ree_data"].empty:
            df = st.session_state["ree_data"]

            # Asegurarse de que el dataframe tiene la columna 'year'
            if 'year' not in df.columns:
                df['year'] = pd.to_datetime(df['datetime']).dt.year

            available_years = sorted(df['year'].unique())

            col1, col2 = st.columns(2)
            with col1:
                year1 = st.selectbox("Selecciona el primer año:", available_years, index=len(available_years) - 2)
            with col2:
                year2 = st.selectbox("Selecciona el segundo año:", available_years, index=len(available_years) - 1)

            if year1 == year2:
                st.warning("Por favor selecciona dos años diferentes para comparar.")
                st.stop()

            years_for_comparison = sorted([year1, year2])

            # Filtrar datos de los años seleccionados
            df_comparison_demanda = df.copy()
            df_filtered_comparison = df_comparison_demanda[
                df_comparison_demanda['year'].isin(years_for_comparison)].copy()

            # Crear 'sort_key' para alinear por día del año
            df_filtered_comparison['sort_key'] = df_filtered_comparison['datetime'].apply(
                lambda dt: dt.replace(year=2000)
            )
            df_filtered_comparison = df_filtered_comparison.sort_values('sort_key')

            # --- Gráfico de Demanda Horaria General Comparativa ---
            fig_comp_hourly = px.line(
                df_filtered_comparison,
                x="sort_key",
                y="value",
                color="year",
                title="Demanda Horaria - Comparativa",
                labels={"sort_key": "Mes y Día", "value": "Demanda (MW)", "year": "Año"},
                hover_data={"year": True, "datetime": "|%Y-%m-%d %H:%M"}
            )
            fig_comp_hourly.update_xaxes(tickformat="%b %d")
            st.plotly_chart(fig_comp_hourly, use_container_width=True)

            # --- Gráficos de Comparación de Métricas Diarias ---
            metrics_comp = df_filtered_comparison.groupby(
                ['year', df_filtered_comparison['datetime'].dt.strftime('%m-%d')])['value'].agg(
                ['mean', 'median', 'min', 'max']).reset_index()

            metrics_comp.columns = ['year', 'month_day', 'mean', 'median', 'min', 'max']
            metrics_comp['sort_key'] = pd.to_datetime('2000-' + metrics_comp['month_day'], format='%Y-%m-%d')
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
                    x="sort_key",
                    y=metric,
                    color="year",
                    title=metric_names[metric],
                    labels={"sort_key": "Fecha (Mes-Día)", metric: "Demanda (MW)", "year": "Año"},
                )
                fig.update_xaxes(tickformat="%b %d")
                st.plotly_chart(fig, use_container_width=True)

            # -----------------------------------
            # Sección de Identificación de Outliers
            # -----------------------------------
            st.subheader("Identificación de Años Outliers (Demanda Anual Total)")

            df_annual_summary = df.groupby('year')['value'].sum().reset_index()
            df_annual_summary.rename(columns={'value': 'total_demand_MW'}, inplace=True)

            if not df_annual_summary.empty and len(df_annual_summary) > 1:
                Q1 = df_annual_summary['total_demand_MW'].quantile(0.25)
                Q3 = df_annual_summary['total_demand_MW'].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                df_annual_summary['is_outlier'] = (
                        (df_annual_summary['total_demand_MW'] < lower_bound) |
                        (df_annual_summary['total_demand_MW'] > upper_bound)
                )

                fig_outliers = px.bar(
                    df_annual_summary,
                    x='year',
                    y='total_demand_MW',
                    color='is_outlier',
                    title='Demanda Total Anual y Años Outliers',
                    labels={'total_demand_MW': 'Demanda Total Anual (MW)', 'year': 'Año', 'is_outlier': 'Es Outlier'},
                    color_discrete_map={False: 'skyblue', True: 'red'}
                )

                st.plotly_chart(fig_outliers, use_container_width=True)

                outlier_years = df_annual_summary[df_annual_summary['is_outlier']]['year'].tolist()

                st.markdown(
                    "**Este gráfico muestra los años identificados como outliers en la demanda total anual.**\n\n"
                    "En este caso, solo se detecta como outlier el año **2025**, lo cual es esperable ya que todavía no ha finalizado "
                    "y su demanda acumulada es significativamente menor.\n\n"
                    "Los años **2022, 2023 y 2024** presentan una demanda anual muy similar, en torno a los **700 MW**, por lo que "
                    "no se consideran outliers según el criterio del rango intercuartílico (IQR)."
                )

                if outlier_years:
                    st.warning(
                        f"Se han identificado los siguientes años como outliers: {', '.join(map(str, outlier_years))}")
                else:
                    st.info("No se han identificado años outliers significativos (según el método IQR).")
            elif not df_annual_summary.empty and len(df_annual_summary) <= 1:
                st.info("Se necesitan al menos 2 años de datos para calcular outliers de demanda anual.")
            else:
                st.warning("No hay datos anuales disponibles para calcular outliers.")
        else:
            st.warning(
                "No hay datos disponibles para la comparación. Realiza primero una consulta en la pestaña de datos.")

    with tab4:
        # -------------------------------------------
        # CONFIGURACIÓN DE STREAMLIT
        # -------------------------------------------
        st.subheader("Predicción de series temporales con modelos de Redes Neuronales Recurrentes")

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
        st.subheader("Predicciones de series temporales con Facebook Prophet")

        # Configuración de granularidades
        granularidades = {
            'Diaria': 'diaria',
            'Semanal': 'semanal',
            'Mensual': 'mensual',
            'Trimestral': 'trimestral',
            'Semestral': 'semestral',
            'Anual': 'anual'
        }

        # Configuración de pasos precalculados
        horizontes = [10, 50, 100]

        # Selección de granularidad y horizonte
        granularidad_seleccionada = st.selectbox("Selecciona la granularidad:", list(granularidades.keys()))
        nombre_granularidad = granularidades[granularidad_seleccionada]

        n_pred = st.selectbox("Selecciona el número de pasos a predecir:", horizontes)

        # Cachear la carga de modelos
        @st.cache_resource
        def load_model(granularidad):
            return joblib.load(f'models/prophet_model_{granularidad}.joblib')

        # Cachear la carga de predicciones
        @st.cache_data
        def load_forecast(granularidad, pasos):
            df = pd.read_csv(f'forecasts/forecast_{granularidad}_{pasos}.csv')
            df['ds'] = pd.to_datetime(df['ds'])
            return df

        # Cargar modelo
        try:
            model_prophet = load_model(nombre_granularidad)
            st.success(f"Modelo {granularidad_seleccionada} cargado correctamente.")
        except Exception as e:
            st.error(f"No se pudo cargar el modelo para {granularidad_seleccionada}: {e}")
            st.stop()

        # Cargar predicción precalculada
        try:
            forecast = load_forecast(nombre_granularidad, n_pred)
            st.success("Predicción cargada correctamente.")
        except Exception as e:
            st.error(f"No se pudo cargar la predicción: {e}")
            st.stop()

        # Mostrar gráfica de predicción
        st.subheader("Predicción")
        fig1 = plot_plotly(model_prophet, forecast)

        # 🔥 Limitar la vista a los últimos 6 meses
        min_date = forecast['ds'].max() - pd.DateOffset(months=6)
        fig1.update_layout(xaxis_range=[min_date, forecast['ds'].max()])

        st.plotly_chart(fig1, use_container_width=True)

        # Mostrar componentes
        st.subheader("Componentes de la predicción")
        fig2 = plot_components_plotly(model_prophet, forecast)
        st.plotly_chart(fig2, use_container_width=True)

    with tab6:
        st.subheader("Gráficos extras de interés")
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

    with tab7:
        st.subheader("Sobre nosotros")

        st.markdown("""Este proyecto ha sido desarrollado a modo de Proyecto Final del Bootcamp de Ciencia de Datos & IA por:""")

        equipo = [
            {
                "nombre": "Adrián Acedo",
                "rol": "Desarrollador | Científico de Datos | Facilitador",
                "github": "https://github.com/AdrianAcedo",
                "linkedin": "https://www.linkedin.com/in/adrianacedoquintanar/",
                "imagen_url": "https://media.licdn.com/dms/image/v2/D4D35AQHvRPQt2he-Ag/profile-framedphoto-shrink_800_800/B4DZXuCJEKHAAg-/0/1743455293952?e=1751054400&v=beta&t=Nsi_PWhEFrUWA7Yvgr4j1nfParqEyEhG9nHAXYff0qk"  # Sustituir por la URL real
            },
            {
                "nombre": "Lucía Varela",
                "rol": "Desarrolladora | Científica de Datos",
                "github": "https://github.com/usuario2",
                "linkedin": "https://linkedin.com/in/usuario2",
                "imagen_url": "https://media.licdn.com/dms/image/v2/D4D03AQGTzePA7mYCCg/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1710544796281?e=1755734400&v=beta&t=EYzHk_ajDGLe20r2PNyVE_ig9R3DcM4xFUhs_P0E1Ps"  # Sustituir por la URL real
            },
            {
                "nombre": "Génesis Rodríguez",
                "rol": "Desarrolladora | Científica de Datos",
                "github": "https://github.com/GenesisSolangel",
                "linkedin": "https://www.linkedin.com/in/g%C3%A9nesis-rodr%C3%ADguez-31a5a6218/",
                "imagen_url": "https://media.licdn.com/dms/image/v2/D4D03AQGTzePA7mYCCg/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1710544796281?e=1755734400&v=beta&t=EYzHk_ajDGLe20r2PNyVE_ig9R3DcM4xFUhs_P0E1Ps"  # Sustituir por la URL real
            }
        ]

        # Mostrar cada miembro en fila
        for persona in equipo:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 30px;">
                    <img src="{persona['imagen_url']}" style="border-radius: 50%; width: 120px; height: 120px; object-fit: cover; margin-right: 20px;">
                    <div>
                        <h3 style="margin-bottom: 5px;">{persona['nombre']}</h3>
                        <p style="margin-bottom: 5px;">{persona['rol']}</p>
                        <p style="margin-bottom: 5px;">GitHub: <a href="{persona['github']}" target="_blank">{persona['github']}</a></p>
                        <p style="margin-bottom: 5px;">LinkedIn: <a href="{persona['linkedin']}" target="_blank">{persona['linkedin']}</a></p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("""Si quieres saber más sobre nuestro trabajo, contáctanos por LinkedIn o consulta nuestro repositorio de GitHub.""")

if __name__ == "__main__":
    main()
