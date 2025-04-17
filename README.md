# ☔ Rainfall Prediction Classifier

Proyecto final de Machine Learning: clasificación binaria para predecir si lloverá hoy en Melbourne, Australia, utilizando datos meteorológicos del día anterior.

## 📁 Estructura del Proyecto

- `rainfall_classifier.py`: script principal con toda la lógica del pipeline y evaluación
- `requirements.txt`: dependencias necesarias para reproducir el entorno
- `.gitignore`: ignora carpetas como `venv/` o archivos temporales

## ⚙️ Requisitos

- Python 3.8+

## 🧪 Instalación

```bash
# Clonar el repositorio
https://github.com/jondalar24/rainfall-prediction-classifier.git
cd rainfall-prediction-classifier

# Crear entorno virtual
python3.10 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```
## ☁️ Descarga del dataset

El archivo `weatherAUS.csv`

## 🚀 Ejecución

```bash
python rainfall_classifier.py
```

## 📊 Objetivo

Entrenar un modelo predictivo usando Random Forest para estimar si lloverá hoy, a partir de datos meteorológicos históricos.

Se incluyen:
- Ingeniería de características (estaciones del año)
- Pipeline con transformación de variables numéricas y categóricas
- Búsqueda de hiperparámetros con GridSearchCV
- Evaluación con métricas de clasificación y matriz de confusión
- Análisis de importancia de variables

## 📌 Autor
Este proyecto está basado en el curso "Machine Learning with Python" de IBM AI Engineering Professional Certificate.
