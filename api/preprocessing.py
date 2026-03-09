import spacy
import re
import pandas as pd

# ==========================================
# 1. CONFIGURACIÓN DEL GANADOR 
# ==========================================
USE_LEMMATIZATION = False 
DROP_STOPWORDS = False
DROP_PUNCTUATION = False     
NORMALIZE_ELONGATION = True  

# ==========================================
# 2. CARGA DEL MOTOR DE SPACY
# ==========================================
# Deshabilitamos 'parser' y 'ner' porque no los necesitamos y hace que la API sea mucho más rápida
print("Cargando motor NLP de spaCy...")
try:
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
except OSError:
    print("Descargando el modelo de spaCy por primera vez...")
    from spacy.cli import download
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])

# ==========================================
# 3. LA FUNCIÓN DE LIMPIEZA CORE
# ==========================================
def limpiar_texto(text: str) -> str:
    """
    Toma un texto crudo enviado por el usuario y le aplica
    exactamente las mismas reglas de la Fase A.
    """
    if pd.isna(text) or not isinstance(text, str): 
        return ""
    
    # 1. Pasar a minúsculas
    text = text.lower()
    
    # 2. Normalizar elongaciones (ej: "heeeellllooo" -> "hello")
    if NORMALIZE_ELONGATION:
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
    # 3. Procesamiento con spaCy
    doc = nlp(text)
    tokens = []
    
    for token in doc:
        # Filtros
        if DROP_STOPWORDS and token.is_stop: 
            continue
        if DROP_PUNCTUATION and token.is_punct: 
            continue
        
        # Lematización o texto original
        word = token.lemma_ if USE_LEMMATIZATION else token.text
        tokens.append(word)
        
    return " ".join(tokens)