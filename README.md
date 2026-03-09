

# Proyecto Sentimientos-memory: Análisis de Sentimientos (Sentimientos403) 

Este repositorio contiene el pipeline completo de Machine Learning Operations (MLOps) para un sistema de clasificación de sentimientos en textos. El proyecto abarca desde la experimentación con técnicas de limpieza de Procesamiento de Lenguaje Natural (NLP) hasta el despliegue automatizado en producción utilizando AWS y FastApi.

##  Arquitectura del Sistema 


[![Arquitectura MLOps](images/AWS-MLFLOW.jpeg)](docs/AWS-MLFLOW.pdf)
 

El flujo de trabajo está diseñado para garantizar la reproducibilidad,
el seguimiento de hiperparámetros y el despliegue automatizado (CI/CD de Modelos).

## Metodología por Fases

El proyecto se desarrolló aplicando el método científico a través de 4 fases experimentales aisladas:

* **Fase A (Ablación y NLP):** Búsqueda de la mejor técnica de limpieza de texto utilizando `spaCy`. Se experimentó activamente con lematización, remoción de stopwords, puntuación y normalización de elongaciones para encontrar la representación óptima.
* **Fase B (Codificación):** Transformación del texto a espacio vectorial evaluando `CountVectorizer` (Bag of Words) frente a `TfidfVectorizer`, combinando pruebas con unigramas y bigramas.
* **Fase C (Modelamiento Clásico):** Implementación de un pipeline de Scikit-Learn que evalúa múltiples algoritmos (Regresión Logística, Random Forest, Multi-Layer Perceptron, KNN). El algoritmo ganador es promovido automáticamente a "Champion" en MLflow y enviado a S3 para su pase a producción.
* **Fase D (Validación vs State-of-the-Art):** El Campeón de Machine Learning clásico se enfrenta en el Test Set intocable contra un modelo de Deep Learning basado en Transformers (`distilbert-base-uncased-finetuned-sst-2-english` de Hugging Face).

##  Resultados y Conclusión Arquitectónica 

El proyecto culminó en una comparativa estricta en el entorno de pruebas (Test Set) priorizando el balance entre precisión matemática y costo computacional (Velocidad de Inferencia).

| Modelo | F1-Score Macro | Tiempo de Inferencia (Test Set) |
| --- | --- | --- |
| ** Regresión Logística (Nuestro Campeón)** | **0.8084** | **4.80 segundos** | 
|  Hugging Face (DistilBERT) | 0.7006 | 7191.28 segundos | 

**Conclusión de Negocio:**
Nuestro pipeline clásico, enriquecido con ingeniería de características (Feature Engineering) y una curación de datos precisa, no solo superó a una Red Neuronal Profunda generalista por más de 10 puntos de F1-Score, sino que demostró ser **~1,500 veces más rápido**. Esto viabiliza un despliegue en producción altamente escalable y de bajo costo en infraestructura de nube, ideal para inferencias en tiempo real mediante nuestra API.

## Stack Tecnológico 

* **Procesamiento de Datos & NLP:** Pandas, PyArrow, spaCy.
* **Machine Learning:** Scikit-Learn (MLP, TF-IDF).
* **Deep Learning:** PyTorch, Transformers (Hugging Face).
* **MLOps & Tracking:** MLflow v3.1+.
* **Cloud Infrastructure:** AWS S3, Amazon EC2, AWS Lambda, AWS Amazon SageMaker AI. 

##  Equipo (NPL) 

* **Líder / MLOps Engineer:** Daniel Varela

---

*Desarrollado para el laboratorio de Procesamiento de Lenguaje Natural.*

---
