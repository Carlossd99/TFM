# 📊 Sistema Integrado de Análisis de Riesgo Dual (Data-Driven)

**Autor:** Carlos Suárez Dávila
**Proyecto:** Trabajo de Fin de Máster (Master en Inteligencia Artificial)

Este repositorio contiene una aplicación financiera híbrida que evalúa el riesgo de activos cotizados combinando tres pilares tecnológicos:
1. **Deep Learning (LSTM):** Red Neuronal Recurrente para predecir el régimen de volatilidad y riesgo a corto plazo (60 días).
2. **Análisis Cuantitativo:** Cálculo de métricas estructurales (CAGR, Sharpe Ratio, Max Drawdown) a largo plazo (5 años).
3. **IA Generativa (Gemini):** Motor LLM que interpreta los datos matemáticos para generar informes pedagógicos automatizados.

---

## ⚙️ 1. Prerrequisitos del Sistema

Antes de clonar el proyecto, asegúrate de tener instaladas las siguientes herramientas en tu sistema:
* **Python 3.11 o 3.12:** (Obligatorio esta versión para garantizar la compatibilidad con TensorFlow).
* **Git:** Para la clonación del repositorio.
* **VS Code (o similar):** Como entorno de desarrollo.

---

## 🔑 2. Obtención de la API Key (Google Gemini)

La aplicación requiere una clave de API para generar los informes de texto.
1. Accede a [Google AI Studio](https://aistudio.google.com/).
2. Inicia sesión con tu cuenta de Google.
3. En el menú lateral, haz clic en **"Get API key"**.
4. Haz clic en **"Create API key"** y cópiala en un lugar seguro. No la compartas con nadie.

---

## 🚀 3. Guía de Instalación y Ejecución Local

Sigue estos pasos rigurosamente en tu terminal para levantar el proyecto desde cero:

### Paso 3.1: Clonar el repositorio
Abre tu terminal (PowerShell, CMD o bash) y ejecuta:
- git clone [https://github.com/Carlossd99/TFM.git]
- cd TFM

### Paso 3.2: Crear el Entorno Virtual
Para aislar las librerías del proyecto del resto de tu ordenador, crea un entorno virtual llamado venv.
- python -m venv venv

### Paso 3.3: Activar el Entorno Virtual
Debes activar el entorno para que las instalaciones se queden dentro del proyecto.
- .\venv\Scripts\Activate.ps1
(Nota: Si PowerShell lanza un error de "Ejecución de scripts deshabilitada", ejecuta este comando primero como Administrador: Set-ExecutionPolicy Unrestricted -Scope CurrentUser, y luego vuelve a intentar la activación).

### Paso 3.4: Instalación de Dependencias
Con el entorno activado, instala todas las librerías necesarias ejecutando:
- pip install -r requirements.txt

### Paso 3.5: Configuración de Variables de Entorno (El archivo .env)
Para que la aplicación se comunique con la IA de forma segura sin exponer tus contraseñas:
Crea un archivo en la raíz del proyecto (junto a app.py) llamado exactamente .env.
Ábrelo y pega tu clave obtenida en el Paso 2 con este formato (sin espacios alrededor del igual):
- GEMINI_API_KEY="AIzaSyTuClaveGeneradaEnGoogleStudio..."

### Paso 3.6: Entrenamiento del Modelo (Opcional)
El repositorio ya incluye un modelo pre-entrenado (modelo_riesgo.keras). Sin embargo, si deseas reentrenar la Red Neuronal LSTM desde cero con datos frescos del mercado:
- python entrenamiento.py
Verás la descarga de datos de 24 activos desde Yahoo Finance y el progreso de entrenamiento por épocas en tu terminal.

### Paso 3.7: Despliegue de la Interfaz Web
Para arrancar la aplicación interactiva, ejecuta:
- streamlit run app.py
Tu navegador predeterminado se abrirá automáticamente en http://localhost:8501 mostrando el sistema de análisis dual.

--- 

## 📁 4. Estructura del Proyecto
- app.py: Archivo principal. Contiene la arquitectura UI/UX y la lógica de integración de modelos.
- entrenamiento.py: Pipeline de descarga de datos, Feature Engineering (magnitudes absolutas, percentiles) y entrenamiento LSTM.
- modelo_riesgo.keras: Objeto serializado que contiene los pesos y sesgos de la red neuronal ya entrenada.
- requirements.txt: Manifiesto estricto de dependencias y librerías de Python.
- .env: Archivo local de variables de entorno (ignorado por Git por seguridad).
- .gitignore: Filtro de seguridad para evitar subidas accidentales de binarios o credenciales.