from pydantic import BaseModel, Field
from typing import List

# ==========================================
# 1. ESQUEMA DE ENTRADA (Input)
# ==========================================
class PredictRequest(BaseModel):
    # Pedimos una lista de strings para soportar tanto una predicción (lista de 1) como lotes (batch)
    textos: List[str] = Field(
        ..., 
        description="Lista de textos para analizar. Para una sola predicción, envíe una lista con un solo elemento.",
        example=["Me encantó el servicio de hoy", "La comida estaba terrible y fría"]
    )

# ==========================================
# 2. ESQUEMAS DE SALIDA (Output)
# ==========================================
class PredictionItem(BaseModel):
    texto_original: str = Field(description="El texto que fue analizado")
    prediccion_cruda: int = Field(description="Valor numérico del modelo (ej. 0 o 1)")
    etiqueta: str = Field(description="Clasificación legible ('Positivo' o 'Negativo')")

class PredictResponse(BaseModel):
    resultados: List[PredictionItem] = Field(description="Lista con los resultados de las predicciones")
    model_version: str = Field(description="Versión del modelo extraída del model_card.json")
    autor_modelo: str = Field(description="Miembro del equipo que entrenó este modelo")