# Imagen base de Python
FROM python:3.9-slim

# Evitar que Python genere archivos .pyc y forzar salida a consola
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias
# Nota: Crea un archivo requirements.txt con: fastapi, uvicorn, pandas, joblib, scikit-learn
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y la carpeta de modelos
COPY . .

# Exponer el puerto de FastAPI
EXPOSE 8000

# Comando para arrancar el motor de la API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]