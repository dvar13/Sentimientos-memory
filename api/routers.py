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
@router.get("/ablation_summary", response_class=HTMLResponse, summary="Reporte visual de Fase A (Desde MLflow)")
async def get_ablation_summary():
    """
    Se conecta a MLflow Tracking, extrae los resultados de los experimentos de la Fase A
    y genera una tabla dinámica, junto con las conclusiones.
    """
    try:
        # 1. Buscar los experimentos en MLflow (Ajusta el nombre al que usaste en Fase A)
        # Nota: Asume que las credenciales de MLflow están cargadas vía .env en el main.py
        runs_df = mlflow.search_runs(experiment_names=["Sentimientos403_FaseA"]) 
        
        # 2. Filtrar y ordenar los datos relevantes (De mayor a menor F1)
        if not runs_df.empty:
            # Seleccionamos solo las columnas que nos importan
            cols_to_keep = [col for col in runs_df.columns if "params" in col or "metrics" in col or "tags.mlflow.runName" in col]
            df_filtered = runs_df[cols_to_keep].copy()
            
            # Limpiamos los nombres de las columnas para que se vean bien en la tabla
            df_filtered.columns = [c.replace("params.", "").replace("metrics.", "").replace("tags.mlflow.runName", "Experimento") for c in df_filtered.columns]
            
            # Ordenamos por la métrica f1 (ajusta el nombre exacto de tu métrica si es diferente)
            if "f1_score" in df_filtered.columns:
                df_filtered = df_filtered.sort_values(by="f1_score", ascending=False)
            
            # Generamos las filas de la tabla HTML dinámicamente
            tabla_html = df_filtered.to_html(index=False, classes="table-mlflow", border=0, justify="center")
        else:
            tabla_html = "<p>No se encontraron experimentos registrados en MLflow para la Fase A.</p>"

    except Exception as e:
        tabla_html = f"<p>Error conectando a MLflow: {str(e)}</p>"

    # 3. Construir el HTML final uniendo la tabla dinámica, la gráfica y las conclusiones estáticas
    html_content = f"""
    <html>
        <head>
            <title>Reporte de Ablación - NLP</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
                table.table-mlflow {{ border-collapse: collapse; width: 80%; margin-bottom: 20px; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }}
                .table-mlflow th, .table-mlflow td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
                .table-mlflow th {{ background-color: #0194E2; color: white; }}
                .table-mlflow tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .table-mlflow tr:hover {{ background-color: #f1f1f1; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #0194E2; padding-bottom: 10px; }}
            </style>
        </head>
        <body>
            <h1>Reporte de Ablación Dinámico (MLflow)</h1>
            
            <h3>Resultados Oficiales de Experimentos</h3>
            {tabla_html}
            
            <h3>Gráfica Comparativa</h3>
            <p><i>Nota: La gráfica se genera a partir de los datos mostrados arriba.</i></p>
            <img src="/static/ablation_plot.png" alt="Gráfica de Ablación" width="700" style="border:1px solid #ccc; border-radius: 5px;"/>
            
            <h3>Conclusiones Analíticas</h3>
            <p><b>Conclusión:</b> Tras analizar los datos extraídos de MLflow, determinamos que la lematización y la remoción de stopwords eliminaban contexto crítico para el análisis de sentimientos. El experimento ganador demostró que conservar la puntuación y normalizar las elongaciones maximiza el F1-Score.</p>
            <p><b>Autor de esta fase:</b> [Nombre del compañero que hizo la Fase A]</p>
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
    # 🚨 Modifica esto con los nombres reales y tareas de tu equipo
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