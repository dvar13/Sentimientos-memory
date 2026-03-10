import mlflow
import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from schemas import PredictRequest, PredictResponse, PredictionItem
from preprocessing import limpiar_texto


# Inicializamos el enrutador
router = APIRouter()

# ==========================================
# 1. INFORMACIÓN DEL MODELO
# ==========================================
@router.get("/model_info", summary="Obtener el Model Card")
async def get_model_info(request: Request):
    """
    Devuelve la descripción y características del modelo en producción
    basado en el archivo model_card.json descargado de S3.
    """
    return request.app.state.model_card

# ==========================================
# 2. INFERENCIA (BATCH & SINGLE)
# ==========================================
@router.post("/predict", response_model=PredictResponse, summary="Predecir Sentimientos")
async def predict_sentiment(payload: PredictRequest, request: Request):
    """
    Recibe una lista de textos (1 o muchos), los limpia usando spaCy
    y devuelve la predicción usando el modelo Campeón.
    """
    textos_crudos = payload.textos
    
    # 1. Limpieza de texto usando la configuración de la Fase A
    textos_limpios = [limpiar_texto(t) for t in textos_crudos]
    
    # 2. Predicción en Batch (Scikit-Learn procesa la lista completa de golpe)
    modelo = request.app.state.model
    predicciones = modelo.predict(textos_limpios)
    
    # 3. Formateo de la respuesta
    resultados = []
    for txt, pred in zip(textos_crudos, predicciones):
        # Asumiendo que 1 = Positivo y 0 = Negativo
        etiqueta = "Positivo" if pred == 1 else "Negativo"
        resultados.append(
            PredictionItem(texto_original=txt, prediccion_cruda=int(pred), etiqueta=etiqueta)
        )
        
    model_card = request.app.state.model_card
    
    return PredictResponse(
        resultados=resultados,
        model_version=model_card.get("version", "unknown"),
        autor_modelo=model_card.get("author", "NPL Team")
    )

# ==========================================
# 3. REPORTE DE ABLACIÓN (Visual HTML)
# ==========================================
@router.get("/ablation_summary", response_class=HTMLResponse, summary="Reporte visual de Fase A")
async def get_ablation_summary():
    """
    Devuelve un reporte visual en HTML con los resultados de la limpieza de datos.
    Versión estática optimizada para bajo consumo de memoria en la nube.
    """
    html_content = """
    <html>
        <head>
            <title>Reporte de Ablación - NLP</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
                table { border-collapse: collapse; width: 80%; margin-bottom: 20px; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
                th { background-color: #0194E2; color: white; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                tr:hover { background-color: #f1f1f1; }
                h1 { color: #2c3e50; border-bottom: 2px solid #0194E2; padding-bottom: 10px; }
            </style>
        </head>
        <body>
            <h1>Reporte de Ablación (Fase A)</h1>
            
            <h3>Resultados de los Experimentos</h3>
            <table>
                <tr>
                    <th>Experimento</th>
                    <th>Lematización</th>
                    <th>StopWords</th>
                    <th>Puntuación</th>
                    <th>Elongación</th>
                    <th>F1-Score (Macro)</th>
                </tr>
                <tr><td>Exp03_Solo_Puntuacion</td><td>Sí</td><td>Sí</td><td>Sí</td><td>No</td><td>0.7736</td></tr>
                <tr><td>Exp10_Punc_Elongacion</td><td>No</td><td>Sí</td><td>Sí</td><td>No</td><td>0.7748</td></tr>
                <tr style="background-color: #d4edda;">
                    <td><b>Exp04_Solo_Elongacionr</b></td>
                    <td><b>No</b></td>
                    <td><b>No</b></td>
                    <td><b>No</b></td>
                    <td><b>Sí</b></td>
                    <td><b>0.7744</b></td>
                </tr>
            </table>
            
            <h3>Gráfica Comparativa</h3>
            <img src="/static/ablation_plot.png" alt="Gráfica de Ablación" width="700" style="border:1px solid #ccc; border-radius: 5px;"/>
            
            <h3>Conclusiones Analíticas</h3>
            <p><b>Conclusión:</b> Tras el proceso de ablación, identificamos que técnicas agresivas como la lematización y la eliminación de stopwords eliminaban contexto semántico valioso (como negaciones). Conservar la puntuación y normalizar elongaciones (ej: "holaaaaa" -> "hola") potenció la capacidad predictiva del modelo, resultando en nuestro ganador.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)
    
# ==========================================
# 4. COMPARATIVA FASE D (ML vs HuggingFace)
# ==========================================
@router.get("/comparison", summary="Comparativa contra State-of-the-Art")
async def get_comparison():
    """
    Devuelve los tiempos de ejecución y F1-Score de nuestro modelo
    contra el modelo DistilBERT de HuggingFace.
    """
    return {
        "dataset_evaluado": "Test Set (Intocable) - 20% de los datos",
        "modelos": {
            "Champion_Clasico": {
                "algoritmo": "MLP + TF-IDF",
                "f1_score_macro": 0.8084,
                "tiempo_train_segundos": 12.5, # Aproximado de tu Fase C
                "tiempo_test_segundos": 4.80
            },
            "HuggingFace_SOTA": {
                "algoritmo": "distilbert-base-uncased-finetuned-sst-2-english",
                "f1_score_macro": 0.7006,
                "tiempo_train_segundos": 0.0, # Zero-Shot pre-entrenado
                "tiempo_test_segundos": 7191.28
            }
        },
        "conclusion": "El modelo Clásico superó al State of the Art en F1-Score por más de 10 puntos al estar adaptado a la jerga del dataset, y demostró ser ~1500 veces más rápido en inferencia."
    }

# ==========================================
# 5. DISTRIBUCIÓN DEL TRABAJO
# ==========================================
@router.get("/work_distribution", summary="Distribución de tareas del equipo")
async def get_work_distribution():
    """
    Tabla de responsabilidades del equipo.
    """
    return {
        "equipo": "NPL",
        "miembros": [
            {
                "nombre": "Daniel Varela",
                "responsabilidades": ["Fase B (Vectorización TF-IDF/BoW)", "Fase D (Hugging Face)", "Arquitectura MLOps AWS y FastAPI"]
            },
            {
                "nombre": "Gustavo Takashi",
                "responsabilidades": ["Fase A (Ablación y spaCy)", "Pipeline de Limpieza"]
            },
            {
                "nombre": "Oscar Guerrero",
                "responsabilidades": ["Fase C (Modelos Clásicos)","Generación de Gráficas y Reportes"]
            },
            {
                "nombre": "Juan Silva",
                "responsabilidades": ["Fase C (Modelos Clásicos)", "Generación de Gráficas y Reportes"]
            }
        ]
    }