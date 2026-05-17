import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv
import google.generativeai as genai

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Analizador Dual de Riesgo IA", page_icon="📈", layout="wide")

# --- OCULTAR ICONOS DE ENLACE DE STREAMLIT (CSS NUCLEAR) ---
st.markdown("""
    <style>
    /* Ocultar el icono de enlace en absolutamente todos los encabezados generados por Markdown */
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
        display: none !important;
    }
    /* Ocultar el contenedor de acciones de los encabezados por si acaso */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Cargar API Key de Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@st.cache_resource
def cargar_modelo_lstm():
    return load_model("modelo_riesgo.keras")

modelo_riesgo = cargar_modelo_lstm()

# --- Encabezado Principal ---
st.title("📊 Sistema de Análisis de Riesgo Dual", anchor=False)
st.subheader("Evaluación de Activos a Corto Plazo (LSTM) y Perspectiva Estructural a Largo Plazo", anchor=False)
st.divider()

# Entrada de usuario
ticker = st.text_input("Introduce el Ticker del activo a analizar (ej. AAPL, NVDA, SPY):", "NVDA").upper()

if ticker:
    with st.spinner(f"Descargando historial completo de 5 años para {ticker}..."):
        # Descargamos los 5 años completos para poder calcular el largo plazo
        datos_completos = yf.download(ticker, period="5y")
        
    if datos_completos.empty:
        st.error("No se han encontrado datos. Por favor, verifica el Ticker introducido.")
    else:
        st.success(f"¡Historial de {ticker} cargado correctamente!")
        
        # Extracción de precios de cierre y retornos de todo el periodo
        cierres_totales = datos_completos['Close'].iloc[:, 0]
        retornos_totales = cierres_totales.pct_change().dropna()
        
        if len(retornos_totales) >= 60:
            
            # ==========================================
            # 1. CÁLCULOS DE LARGO PLAZO (ESTRUCTURAL - 5 AÑOS)
            # ==========================================
            precio_actual = float(cierres_totales.iloc[-1])
            precio_hace_5y = float(cierres_totales.iloc[0])
            
            # CAGR (Tasa de Crecimiento Anual Compuesto)
            cagr = ((precio_actual / precio_hace_5y) ** (1 / 5) - 1) * 100
            
            # Volatilidad Anualizada Histórica (5 años)
            volatilidad_5y = float(retornos_totales.std() * np.sqrt(252) * 100)
            
            # Ratio de Sharpe Histórico (Asumiendo tasa libre de riesgo = 0% para simplicidad)
            retorno_anualizado_medio = retornos_totales.mean() * 252 * 100
            sharpe_5y = retorno_anualizado_medio / volatilidad_5y
            
            # Max Drawdown Histórico (5 años)
            picos_5y = cierres_totales.cummax()
            drawdowns_5y = (cierres_totales - picos_5y) / picos_5y
            max_drawdown_5y = float(drawdowns_5y.min() * 100)
            
            # ==========================================
            # 2. CÁLCULOS DE CORORTO PLAZO (VENTANA DE 60 DÍAS)
            # ==========================================
            precios_60d = cierres_totales[-60:]
            retornos_60d = retornos_totales[-60:].values
            
            # Preparación del tensor 3D de magnitud absoluta para nuestra LSTM
            X_input = np.abs(retornos_60d).reshape((1, 60, 1))
            
            # Volatilidad e impactos recientes
            volatilidad_60d = float(retornos_60d.std() * np.sqrt(252) * 100)
            precio_inicio_60d = float(precios_60d.iloc[0])
            rendimiento_60d = ((precio_actual - precio_inicio_60d) / precio_inicio_60d) * 100
            
            # Predicción del régimen de riesgo con la Red Neuronal
            prediccion_probs = modelo_riesgo.predict(X_input, verbose=0)
            nivel_riesgo_lstm = int(np.argmax(prediccion_probs[0]))
            probs_barras = prediccion_probs[0][1:] * 100
            
            # ==========================================
            # 3. INTERFAZ GRÁFICA MEDIANTE PESTAÑAS (TABS)
            # ==========================================
            tab_corto, tab_largo = st.tabs(["⚡ Corto Plazo (Régimen LSTM)", "🏛️ Largo Plazo (Estructural)"])
            
            with tab_corto:
                st.subheader("Análisis de Riesgo Inmediato (Últimos 60 días hábiles)", anchor=False)
                col_c1, col_c2 = st.columns([2, 1])
                
                with col_c1:
                    st.line_chart(precios_60d)
                    metrics_c = st.columns(3)
                    metrics_c[0].metric("Precio Actual", f"${precio_actual:.2f}")
                    metrics_c[1].metric("Rendimiento Reciente", f"{rendimiento_60d:.2f}%")
                    metrics_c[2].metric("Volatilidad Corto Plazo", f"{volatilidad_60d:.2f}%")
                    
                with col_c2:
                    # Cambiado de markdown a subheader para forzar el anchor=False
                    st.subheader(f"Veredicto de Riesgo LSTM: **Nivel {nivel_riesgo_lstm} / 5**", anchor=False)
                    st.write("Distribución de probabilidad del modelo:")
                    df_probs = pd.DataFrame({
                        "Nivel de Riesgo": ["1 (M. Bajo)", "2 (Bajo)", "3 (Medio)", "4 (Alto)", "5 (M. Alto)"],
                        "Confianza (%)": probs_barras
                    }).set_index("Nivel de Riesgo")
                    st.bar_chart(df_probs)
            
            with tab_largo:
                st.subheader("Comportamiento Estructural de Fondo (Historial de 5 años)", anchor=False)
                col_l1, col_l2 = st.columns([2, 1])
                
                with col_l1:
                    st.line_chart(cierres_totales)
                    
                with col_l2:
                    st.subheader("Métricas de Eficiencia e Impacto", anchor=False)
                    st.metric("CAGR (Crecimiento Anual Compuesto)", f"{cagr:.2f}%")
                    st.metric("Ratio de Sharpe (Eficiencia)", f"{sharpe_5y:.2f}")
                    st.metric("Máxima Caída Histórica (5 años)", f"{max_drawdown_5y:.2f}%")
                    st.metric("Volatilidad Histórica Media", f"{volatilidad_5y:.2f}%")

            # ==========================================
            # 4. MOTOR PEDAGÓGICO DE GEMINI
            # ==========================================
            st.divider()
            st.subheader("👨‍🏫 Informe de Horizonte Temporal del Analista IA", anchor=False)
            
            prompt = f"""
            Actúa como un asesor financiero cuantitativo de élite. Escribe un informe educativo detallado sobre {ticker}.
            
            DATOS DE CORTO PLAZO (60 DÍAS):
            - Volatilidad reciente: {volatilidad_60d:.2f}%
            - Rendimiento reciente: {rendimiento_60d:.2f}%
            - Riesgo asignado por la Red Neuronal LSTM: Nivel {nivel_riesgo_lstm}/5
            
            DATOS DE LARGO PLAZO (5 AÑOS):
            - Crecimiento Anual Compuesto (CAGR): {cagr:.2f}%
            - Ratio de Sharpe (Eficiencia): {sharpe_5y:.2f}
            - Caída Máxima Histórica (Max Drawdown): {max_drawdown_5y:.2f}%
            - Volatilidad Histórica Estructural: {volatilidad_5y:.2f}%
            
            INSTRUCCIONES PARA EL INFORME:
            1. En el primer párrafo, analiza el comportamiento de corto plazo explicando qué significa el nivel de riesgo asignado por la LSTM en base a la volatilidad reciente.
            2. En el segundo párrafo, introduce la perspectiva de largo plazo. Explica la paradoja de cómo un activo puede ser de riesgo muy alto a corto plazo (Nivel 4 o 5) pero presentar un crecimiento (CAGR) o eficiencia (Sharpe) excelentes si se mantiene en el tiempo.
            3. Usa un tono académico, claro y estrictamente pedagógico. Destaca términos financieros clave en negrita. No des recomendaciones de compra o venta.
            """
            
            with st.spinner("Generando análisis pedagógico de doble horizonte..."):
                try:
                    # NOTA: Asegúrate de usar gemini-3-flash o gemini-1.5-flash según la versión de tu API
                    model = genai.GenerativeModel('gemini-2.5-flash') 
                    response = model.generate_content(prompt)
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Error de comunicación con la API de Gemini: {e}")
                    
        else:
            st.warning("El activo no dispone de suficiente historial para el análisis dual.")