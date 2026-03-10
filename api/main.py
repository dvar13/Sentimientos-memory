import os
import json
import joblib
import boto3
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import routers

# 1. Cargar variables de entorno (Ocultando nuestros secretos)
load_dotenv()
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Ajusta esto al nombre real de tu archivo .pkl guardado en la Fase C
MODEL_FILE_NAME = "model.pkl" 
S3_MODEL_CARD = "data/produccion/Modelos"
S3_MODEL_PREFIX = "data/produccion/Modelos/tmp_champion_model"

# ==========================================
# 2. EVENTO DE INICIO (Lifespan)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Se ejecuta una sola vez cuando el servidor arranca.
    Descarga el modelo de S3 y lo deja guardado en la memoria RAM (Caché).
    """
    print(f" Iniciando servidor. Conectando al bucket S3: {S3_BUCKET_NAME}...")
    
    # Iniciamos cliente de AWS
    s3_client = boto3.client("s3")
    
    try:
        # A. Descargar y Cargar el Model Card
        print("Descargando model_card.json...")
        s3_client.download_file(S3_BUCKET_NAME, f"{S3_MODEL_CARD}/model_card.json", "model_card.json")
        with open("model_card.json", "r") as f:
            app.state.model_card = json.load(f)
            
        # B. Descargar y Cargar el Modelo .pkl
        print(f"Descargando {MODEL_FILE_NAME}...")
        s3_client.download_file(S3_BUCKET_NAME, f"{S3_MODEL_PREFIX}/{MODEL_FILE_NAME}", "local_model.pkl")
        app.state.model = joblib.load("local_model.pkl")
        
        print("Modelo cargado en memoria exitosamente y listo para inferencias!")
        
    except Exception as e:
        print(f"Error crítico al descargar desde S3: {str(e)}")
        print("Asegúrate de que tu bucket, rutas y variables de AWS estén correctamente configurados.")
        # Opcional: Podrías detener el servidor aquí si el modelo es estrictamente necesario
        app.state.model = None
        app.state.model_card = {}

    yield # Aquí el servidor se queda corriendo, atendiendo peticiones...
    
    # Cuando apagues el servidor, se ejecuta esta limpieza
    print(" Apagando el servidor. Limpiando memoria...")
    if os.path.exists("model_card.json"): os.remove("model_card.json")
    if os.path.exists("local_model.pkl"): os.remove("local_model.pkl")

# ==========================================
# 3. INSTANCIA DE LA APLICACIÓN
# ==========================================
app = FastAPI(
    title="API de Sentimientos (NPL Team)",
    description="Microservicio MLOps para clasificación de texto, conectado a MLflow y S3",
    version="1.0.0",
    lifespan=lifespan
)

# ==========================================
# 4. CONFIGURACIÓN DE ARCHIVOS ESTÁTICOS Y RUTAS
# ==========================================
# Montamos la carpeta "static" para que FastAPI pueda mostrar la gráfica de la Fase A
app.mount("/static", StaticFiles(directory="static"), name="static")

# Conectamos todos los endpoints de nuestro archivo routers.py
app.include_router(routers.router)

# ==========================================
# 5. HEALTH CHECK (Ruta Raíz)
# ==========================================
@app.get("/", summary="Health Check")
async def health_check():
    """
    Endpoint básico para verificar que la API está viva.
    AWS suele usar esta ruta para monitorear el microservicio.
    """
    return {
        "status": "online",
        "mensaje": "Bienvenido a la API de Sentimientos403",
        "docs_url": "/docs"
    }