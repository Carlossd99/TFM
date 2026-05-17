import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

tickers = [
    'BIL', 'SHY', 'TLT', 'IEF', 'SPY', 'DIA', 'GLD', 'JNJ', 'KO', 
    'QQQ', 'GOOGL', 'MSFT', 'V', 'JPM', 'AAPL', 'AMZN', 'NFLX', 
    'META', 'DIS', 'TSLA', 'NVDA', 'COIN', 'ARKK', 'MSTR'
]

print("1. Descargando 5 años de historia...")
datos_cierre = yf.download(tickers, period="5y")['Close'].dropna()
retornos_totales = datos_cierre.pct_change().dropna()

print("\n2. Asignando Riesgo (1-5) mediante Percentiles Estadísticos...")
volatilidades = retornos_totales.std() * np.sqrt(252)
df_clasificacion = pd.DataFrame(volatilidades, columns=['Volatilidad'])
df_clasificacion['Riesgo'] = pd.qcut(df_clasificacion['Volatilidad'], q=5, labels=[1, 2, 3, 4, 5])
etiquetas_dinamicas = df_clasificacion['Riesgo'].to_dict()

print("\n3. Creando ventanas de 60 días (Magnitud Absoluta)...")
X, y = [], []
VENTANA = 60
retornos_absolutos = np.abs(retornos_totales) # Clave para que la LSTM entienda la volatilidad

for ticker, nivel_riesgo in etiquetas_dinamicas.items():
    datos_empresa = retornos_absolutos[ticker].values
    for i in range(len(datos_empresa) - VENTANA):
        X.append(datos_empresa[i : i + VENTANA])
        y.append(nivel_riesgo)

X = np.array(X).reshape((-1, VENTANA, 1))
y = to_categorical(np.array(y), num_classes=6)

print("\n4. Entrenando Oráculo de Riesgo LSTM...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = Sequential([
    LSTM(64, return_sequences=True, input_shape=(VENTANA, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(6, activation='softmax') # Volvemos a probabilidad repartida (Riesgo 1 al 5)
])

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=1)

perdida, precision = modelo.evaluate(X_test, y_test, verbose=0)
print(f"\n=== Precisión de Clasificación de Riesgo: {precision * 100:.2f}% ===")
modelo.save("modelo_riesgo.keras")