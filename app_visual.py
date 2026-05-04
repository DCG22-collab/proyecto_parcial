import streamlit as st
import requests

st.set_page_config(page_title="Analizador Instagram CRISP-DM", page_icon="📸")

st.title("📸 Analizador de Salud Mental - Instagram")
st.markdown("Basado en el modelo de **Inferencia en Cascada** del proyecto.")

# Formulario sincronizado con la API
with st.form("perfil_usuario"):
    st.subheader("Información del Perfil")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Edad", 13, 100, 18)
        gender = st.selectbox("Género", ["Female", "Male", "Non-binary", "Prefer not to say"])
        urban_rural = st.selectbox("Entorno", ["Urban", "Rural", "Suburban"])
        income = st.selectbox("Nivel de Ingresos", ["High", "Low", "Lower-middle", "Middle", "Upper-middle"])
        edu = st.selectbox("Educación", ["Bachelor’s", "High school", "Master’s", "Other", "PhD", "Some college"])
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
    # Preparación del payload para la API
    # Calculamos valores derivados para no pedirle todo al usuario
    payload = {
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
    }
    
    try:
        # Nota: 'api' es el nombre del servicio si usas Docker Compose
        response = requests.post("http://localhost:8000/predecir", json=payload)
        data = response.json()
        
        if data["status"] == "success":
            res = data["predicciones"]
            st.success("¡Análisis completado!")
            r1, r2, r3 = st.columns(3)
            r1.metric("Nivel de Estrés", f"{res['nivel_estres_percibido']}")
            r2.info(f"Riesgo: {res['etiqueta_riesgo']}")
            r3.metric("Felicidad Proyectada", f"{res['nivel_felicidad_proyectado']}")
        else:
            st.error(f"Error en la API: {data['message']}")
            
    except Exception as e:
        st.error(f"No se pudo conectar con la API. Asegúrate de que esté corriendo. {e}")