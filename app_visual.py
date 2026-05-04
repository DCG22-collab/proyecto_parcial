import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Analizador Instagram CRISP-DM", page_icon="📸")

# Carga de modelos directamente (sin API)
@st.cache_resource
def cargar_modelos():
    BASE_DIR = Path(__file__).resolve().parent
    MODELOS_DIR = BASE_DIR / "modelos_despliegue"

    modelo_estres    = joblib.load(MODELOS_DIR / "modelo_estres_final.pkl")
    modelo_riesgo    = joblib.load(MODELOS_DIR / "modelo_clasificacion_final.pkl")
    modelo_felicidad = joblib.load(MODELOS_DIR / "modelo_felicidad_final.pkl")
    scaler_base      = joblib.load(MODELOS_DIR / "scaler_base.pkl")
    scaler_felicidad = joblib.load(MODELOS_DIR / "scaler_felicidad.pkl")

    cols_base     = list(modelo_estres.feature_names_in_)
    cols_happ     = list(modelo_felicidad.feature_names_in_)
    cols_num_base = list(scaler_base.feature_names_in_)
    cols_num_happ = list(scaler_felicidad.feature_names_in_)

    return modelo_estres, modelo_riesgo, modelo_felicidad, scaler_base, scaler_felicidad, cols_base, cols_happ, cols_num_base, cols_num_happ

modelo_estres, modelo_riesgo, modelo_felicidad, scaler_base, scaler_felicidad, cols_base, cols_happ, cols_num_base, cols_num_happ = cargar_modelos()

st.title("📸 Analizador de Salud Mental - Instagram")
st.markdown("Basado en el modelo de **Inferencia en Cascada** del proyecto.")

with st.form("perfil_usuario"):
    st.subheader("Información del Perfil")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Edad", 13, 100, 18)
        gender = st.selectbox("Género", ["Female", "Male", "Non-binary", "Prefer not to say"])
        urban_rural = st.selectbox("Entorno", ["Urban", "Rural", "Suburban"])
        income = st.selectbox("Nivel de Ingresos", ["High", "Low", "Lower-middle", "Middle", "Upper-middle"])
        edu = st.selectbox("Educación", ["Bachelor's", "High school", "Master's", "Other", "PhD", "Some college"])
        status = st.selectbox("Estatus Laboral", ["Freelancer", "Full-time employed", "Part-time", "Retired", "Student", "Unemployed"])

    with col2:
        sleep = st.number_input("Horas de sueño diarias", 0.0, 24.0, 7.0)
        exercise = st.number_input("Horas ejercicio/semana", 0.0, 100.0, 5.0)
        work = st.number_input("Horas trabajo/semana", 0.0, 120.0, 40.0)
        events = st.number_input("Eventos sociales/mes", 0, 100, 4)
        resp_rate = st.slider("Tasa respuesta notificaciones", 0.0, 1.0, 0.5)

    st.subheader("Hábitos en Instagram")
    c3, c4 = st.columns(2)
    with c3:
        daily_min = st.number_input("Minutos totales diarios", 0.0, 1440.0, 60.0)
        sessions = st.number_input("Sesiones al día", 1, 500, 5)
        posts = st.number_input("Posts creados/semana", 0, 500, 1)
    with c4:
        reels_day = st.number_input("Reels vistos al día", 0, 1000, 20)
        pref = st.selectbox("Preferencia de Contenido", ["Live", "Mixed", "Photos", "Reels", "Stories", "Videos"])
        time_reels = st.number_input("Minutos en Reels al día", 0.0, daily_min, daily_min * 0.5)

    submit = st.form_submit_button("Realizar Predicción")

if submit:
    try:
        # A. Construir DataFrame
        datos_crudos = pd.DataFrame([{
            "age": age, "gender": gender, "urban_rural": urban_rural,
            "income_level": income, "education_level": edu,
            "employment_status": status, "sleep_hours_per_night": sleep,
            "exercise_hours_per_week": exercise, "weekly_work_hours": work,
            "social_events_per_month": events, "daily_active_minutes_instagram": daily_min,
            "sessions_per_day": sessions,
            "average_session_length_minutes": daily_min / sessions,
            "notification_response_rate": resp_rate, "posts_created_per_week": posts,
            "reels_watched_per_day": reels_day,
            "content_type_preference": pref,
            "time_on_feed_per_day": daily_min - time_reels,
            "time_on_reels_per_day": time_reels
        }])

        # B. Calcular Generation
        bins_edad  = [13, 26, 42, 58, 100]
        labels_gen = ["Gen Z", "Millennials", "Gen X", "Boomers"]
        datos_crudos["Generation"] = pd.cut(
            datos_crudos["age"], bins=bins_edad, labels=labels_gen, include_lowest=True
        )

        # C. One-Hot Encoding
        datos_transformados = pd.get_dummies(datos_crudos)

        # D. Predicción de estrés
        X_base = datos_transformados.reindex(columns=cols_base, fill_value=0)
        X_base[cols_num_base] = scaler_base.transform(X_base[cols_num_base])
        estres_pred = float(modelo_estres.predict(X_base)[0])
        riesgo_pred = str(modelo_riesgo.predict(X_base)[0])

        # E. Predicción de felicidad
        X_happ = datos_transformados.reindex(columns=cols_happ, fill_value=0)
        X_happ["perceived_stress_score"] = estres_pred
        X_happ[cols_num_happ] = scaler_felicidad.transform(X_happ[cols_num_happ])
        felicidad_pred = float(modelo_felicidad.predict(X_happ)[0])

        st.success("¡Análisis completado!")
        r1, r2, r3 = st.columns(3)
        r1.metric("Nivel de Estrés", f"{round(estres_pred, 2)}")
        r2.info(f"Riesgo: {riesgo_pred}")
        r3.metric("Felicidad Proyectada", f"{round(felicidad_pred, 2)}")

    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")