import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import plotly.express as px
from supabase import create_client, Client
import schedule
import threading
import time as tiempo
import uuid
import requests
from dotenv import load_dotenv
import os

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

# Configuración de credenciales de Supabase
#SUPABASE_URL = "https://gbfxqkzjzamqlqhzvbqc.supabase.co"  # Sustituye con tu URL
#SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdiZnhxa3pqemFtcWxxaHp2YnFjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDkwMjc3MDksImV4cCI6MjA2NDYwMzcwOX0.ju_muEo9aTGT8FWFYpP-5_uEaywdSn7xOPllt1VQtUQ"  # Sustituye con tu anon key
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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


# Función de extracción de datos de los últimos x años, devuelve un DataFrame de Pandas
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

            # Iteramos sobre cada endpoint
            for name, (path, granularity) in ENDPOINTS.items():
                params = {
                    "start_date": month_start.strftime("%Y-%m-%dT%H:%M"),
                    "end_date": end_date_for_request.strftime("%Y-%m-%dT%H:%M"),
                    "geo_trunc": "electric_system",
                    "geo_limit": "peninsular",
                    "geo_ids": "8741"
                }

                month_data = get_data(name, (path, granularity), params)

                # Y sacamos los datos
                if month_data:
                    df = pd.DataFrame(month_data)
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
                    df['endpoint'] = name
                    df['extraction_timestamp'] = datetime.utcnow()
                    df['record_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

                    df = df[['record_id', 'value', 'percentage', 'datetime',
                             'primary_category', 'sub_category', 'year', 'month',
                             'day', 'hour', 'endpoint', 'extraction_timestamp']]

                    all_dfs.append(df)

                tiempo.sleep(1)

    # ---------------- SUPABASE: Conexión e Inserción ---------------- #
    from supabase import create_client, Client



    # 📁 Diccionario tabla → DataFrame
    # tablas_dfs = {
    #    "demanda": df_demanda,
    #    "balance": df_balance,
    #    "generacion": df_generacion,
    #    "intercambios": df_intercambios,
    #    "intercambios_baleares": df_intercambios_baleares
    # }

    # para insertar cada DataFrame
    def insertar_en_supabase(nombre_tabla, df):
        import uuid

        df = df.copy()

        # Asegurar UUID únicos
        df["record_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        # Convertir fechas a string ISO
        for col in ["datetime", "extraction_timestamp"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Reemplazar NaN por None
        df = df.where(pd.notnull(df), None)

        # Convertir a lista de diccionarios e insertar
        data = df.to_dict(orient="records")

        try:
            response = supabase.table(nombre_tabla).insert(data).execute()
            print(f"✅ Insertados en '{nombre_tabla}': {len(data)} filas")
        except Exception as e:
            print(f"❌ Error al insertar en '{nombre_tabla}': {e}")

    # ------------------ ACTUALIZACIÓN AUTOMÁTICA DESDE API + SUPABASE ------------------

    def actualizar_datos_desde_api():
        print(f"[{datetime.now()}] ⏳ Ejecutando extracción desde API...")

        try:
            current_date = datetime.now()
            start_date = current_date - timedelta(days=2)  # Solo últimos 2 días para no sobrecargar

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
                    df['endpoint'] = name
                    df['extraction_timestamp'] = datetime.utcnow()
                    df['record_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

                    df = df[['record_id', 'value', 'percentage', 'datetime',
                             'primary_category', 'sub_category', 'year', 'month',
                             'day', 'hour', 'endpoint', 'extraction_timestamp']]

                    insertar_en_supabase(name, df)
                    print(f"✅ Datos de '{name}' actualizados.")
                    tiempo.sleep(1)
                else:
                    print(f"⚠️ No se obtuvieron datos de '{name}'")

            print("✅ Actualización completa.")
        except Exception as e:
            print("❌ Error durante la actualización automática desde API:", e)

    def iniciar_programador_api():
        schedule.every(1).hours.do(actualizar_datos_desde_api)

        while True:
            schedule.run_pending()
            tiempo.sleep(60)

    # 🧵 Lanza el hilo para ejecución en background
    threading.Thread(target=iniciar_programador_api, daemon=True).start()

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


st.set_page_config(page_title="Red Eléctrica", layout="centered")

def get_data_from_supabase(table_name, start_date, end_date, page_size=1000):
    end_date += timedelta(days=1)
    start_iso = start_date.isoformat()
    end_iso = end_date.isoformat()

    all_data = []
    offset = 0

    while True:
        response = (
            supabase
            .table(table_name)
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

# ----------------------------- INTERFAZ -----------------------------

def main():
    st.title("Análisis de la Red Eléctrica Española")

    tab1, tab2, tab3 = st.tabs(["Descripción", "Consulta de datos", "Visualización"])

    # Tab 1: Descripción
    with tab1:
        st.subheader("¿Qué es esta app?")
        st.markdown("""
        Esta aplicación se conecta con la base de datos en Supabase que contiene datos históricos de la Red Eléctrica Española. 
        Permite consultar y visualizar datos de demanda, balance, generación e intercambios mediante filtros por fechas y categorías.
        """)

    # Tab 2: Consulta de datos
    with tab2:
        st.subheader("Consulta de datos")

        modo = st.radio("Tipo de consulta:", ["Últimos días", "Año específico"], horizontal=True)

        if modo == "Últimos días":
            dias = st.selectbox("¿Cuántos días atrás?", [7, 14, 30])
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=dias)
        else:
            current_year = datetime.now().year
            years = [current_year - i for i in range(0, 3)]  # [2025, 2024, 2023]
            año = st.selectbox("Selecciona el año a consultar:", years)
            if año == datetime.now().year:
                start_date = datetime(año, 1, 1, tzinfo=timezone.utc)
                end_date = datetime.now(timezone.utc)
            else:
                start_date = datetime(año, 1, 1, tzinfo=timezone.utc)
                end_date = datetime(año, 12, 31, 23, 59, tzinfo=timezone.utc)

        table = st.selectbox("Selecciona la tabla que deseas consultar:", [
            "demanda", "balance", "generacion", "intercambios", "intercambios_baleares"
        ])

        with st.spinner("Consultando Supabase..."):
            df = get_data_from_supabase(table, start_date, end_date)

        if not df.empty:
            st.session_state["ree_data"] = df
            st.session_state["tabla"] = table
            st.session_state["start_date"] = start_date
            st.session_state["end_date"] = end_date
            st.write(f"Datos recuperados: {len(df)} filas")
            st.write("Último dato:", df['datetime'].max())
            st.success("Datos cargados correctamente desde Supabase.")
        else:
            st.warning("No se encontraron datos para ese período.")

    # Tab 3: Visualización
    with tab3:
        st.subheader("Visualización")

        if "ree_data" in st.session_state:
            df = st.session_state["ree_data"]

            if table == "demanda":
                fig = px.area(df, x="datetime", y="value", title="Demanda eléctrica", labels={"value": "Demanda (MW)"})
            elif table == "balance":
                fig = px.bar(df, x="datetime", y="value", color="primary_category", barmode="group",
                             title="Balance eléctrico")
            elif table == "generacion":
                fig = px.line(df, x="datetime", y="value", color="primary_category", title="Generación")
            elif table == "intercambios":
                fig = px.line(df, x="datetime", y="value", color="sub_category", title="Intercambios por frontera")
            elif table == "intercambios_baleares":
                fig = px.line(df, x="datetime", y="value", title="Intercambios con Baleares")
            else:
                fig = px.line(df, x="datetime", y="value", title="Visualización general")

            st.plotly_chart(fig, use_container_width=True)



            with st.expander("Ver datos en tabla"):
                st.dataframe(df, use_container_width=True)
        else:
            st.info("Primero consulta los datos desde la pestaña anterior.")


if __name__ == "__main__":
    main()
