from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from pathlib import Path
from typing import Literal

# 1. Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Predicción - Salud Mental en Instagram",
    description="Sistema desplegado bajo la metodología CRISP-DM.",
    version="2.0.0"
)

# 2. Carga global de modelos y scalers
BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "modelos_despliegue"

try:
    modelo_estres    = joblib.load(MODELOS_DIR / "modelo_estres_final.pkl")
    modelo_riesgo    = joblib.load(MODELOS_DIR / "modelo_clasificacion_final.pkl")
    modelo_felicidad = joblib.load(MODELOS_DIR / "modelo_felicidad_final.pkl")

    # scaler_base: estrés y clasificación (13 variables digitales)
    scaler_base      = joblib.load(MODELOS_DIR / "scaler_base.pkl")

    # scaler_felicidad: felicidad (13 variables digitales + perceived_stress_score)
    scaler_felicidad = joblib.load(MODELOS_DIR / "scaler_felicidad.pkl")

    # Orden de columnas extraído directamente de los modelos entrenados
    cols_base = list(modelo_estres.feature_names_in_)
    cols_happ = list(modelo_felicidad.feature_names_in_)

    # Columnas numéricas que cada scaler conoce
    cols_num_base = list(scaler_base.feature_names_in_)
    cols_num_happ = list(scaler_felicidad.feature_names_in_)

    print("✅ Modelos y Scalers cargados correctamente.")
    print(f"   cols_happ     : {len(cols_happ)} columnas")
    print(f"   cols_num_base : {cols_num_base}")
    print(f"   cols_num_happ : {cols_num_happ}")

except Exception as e:
    print(f"❌ Error al iniciar el servidor: {e}")


# 3. Esquema de Entrada del Usuario
class UsuarioInput(BaseModel):
    age: int = Field(..., ge=13, le=100)
    gender: Literal["Female", "Male", "Non-binary", "Prefer not to say"]
    urban_rural: Literal["Rural", "Suburban", "Urban"]
    income_level: Literal["High", "Low", "Lower-middle", "Middle", "Upper-middle"]
    education_level: Literal["Bachelor\u2019s", "High school", "Master\u2019s", "Other", "PhD", "Some college"]
    employment_status: Literal["Freelancer", "Full-time employed", "Part-time", "Retired", "Student", "Unemployed"]

    sleep_hours_per_night: float = Field(..., ge=0.0, le=24.0)
    exercise_hours_per_week: float = Field(..., ge=0.0, le=100.0)
    weekly_work_hours: float = Field(..., ge=0.0, le=120.0)
    social_events_per_month: int = Field(..., ge=0, le=100)

    daily_active_minutes_instagram: float = Field(..., ge=0.0, le=1440.0)
    sessions_per_day: int = Field(..., ge=0, le=500)
    average_session_length_minutes: float = Field(..., ge=0.0, le=1440.0)
    notification_response_rate: float = Field(..., ge=0.0, le=1.0)
    posts_created_per_week: int = Field(..., ge=0, le=500)
    reels_watched_per_day: int = Field(..., ge=0, le=1000)
    content_type_preference: Literal["Live", "Mixed", "Photos", "Reels", "Stories", "Videos"]
    time_on_feed_per_day: float = Field(..., ge=0.0, le=1440.0)
    time_on_reels_per_day: float = Field(..., ge=0.0, le=1440.0)


# 4. Endpoint de Salud
@app.get("/")
def home():
    return {"mensaje": "API de Predicción CRISP-DM Activa 🚀"}


# 5. Endpoint de Predicción
@app.post("/predecir")
def predecir_perfil(usuario: UsuarioInput):
    try:
        # A. Convertir entrada a DataFrame
        datos_crudos = pd.DataFrame([usuario.model_dump()])

        # B. Calcular Generation — mismos bins que el notebook corregido
        bins_edad  = [13, 26, 42, 58, 100]
        labels_gen = ["Gen Z", "Millennials", "Gen X", "Boomers"]
        datos_crudos["Generation"] = pd.cut(
            datos_crudos["age"], bins=bins_edad, labels=labels_gen, include_lowest=True
        )

        # C. One-Hot Encoding
        datos_transformados = pd.get_dummies(datos_crudos)

        # D. Alinear columnas para estres y riesgo
        X_base = datos_transformados.reindex(columns=cols_base, fill_value=0)

        # E. Escalar con scaler_base (mismo que se uso en entrenamiento de estres/riesgo)
        X_base[cols_num_base] = scaler_base.transform(X_base[cols_num_base])

        # F. Prediccion de estres y riesgo
        estres_pred = float(modelo_estres.predict(X_base)[0])
        riesgo_pred = str(modelo_riesgo.predict(X_base)[0])

        # G. Preparar features para felicidad + anadir estres predicho
        X_happ = datos_transformados.reindex(columns=cols_happ, fill_value=0)
        X_happ["perceived_stress_score"] = estres_pred

        # H. Escalar con scaler_felicidad (incluye perceived_stress_score)
        X_happ[cols_num_happ] = scaler_felicidad.transform(X_happ[cols_num_happ])

        # I. Prediccion de felicidad
        felicidad_pred = float(modelo_felicidad.predict(X_happ)[0])

        return {
            "status": "success",
            "predicciones": {
                "nivel_estres_percibido":     round(estres_pred, 2),
                "etiqueta_riesgo":            riesgo_pred,
                "nivel_felicidad_proyectado": round(felicidad_pred, 2)
            }
        }

    except Exception as e:
        return {"status": "error", "message": f"Error en procesamiento: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)