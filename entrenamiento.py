import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

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
retornos_absolutos = np.abs(retornos_totales)

for ticker, nivel_riesgo in etiquetas_dinamicas.items():
    datos_empresa = retornos_absolutos[ticker].values
    for i in range(len(datos_empresa) - VENTANA):
        X.append(datos_empresa[i : i + VENTANA])
        y.append(nivel_riesgo)

X = np.array(X).reshape((-1, VENTANA, 1))
y = to_categorical(np.array(y), num_classes=6)

print("\n4. Entrenando modelo LSTM")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

modelo = Sequential([
    LSTM(32, return_sequences=True, input_shape=(VENTANA, 1)),
    Dropout(0.2),
    LSTM(16, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(6, activation='softmax')
])

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
parada_temprana = EarlyStopping(
    monitor='val_accuracy', 
    patience=4, 
    restore_best_weights=True
)
historial =modelo.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1, callbacks=[parada_temprana])

perdida, precision = modelo.evaluate(X_test, y_test, verbose=0)
print(f"\n=== Precisión de Clasificación de Riesgo: {precision * 100:.2f}% ===")
modelo.save("modelo_riesgo.keras")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. Gráfica de Curvas de Aprendizaje (Loss y Accuracy)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(historial.history['loss'], label='Pérdida (Entrenamiento)')
plt.plot(historial.history['val_loss'], label='Pérdida (Validación)')
plt.title('Curva de Pérdida (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (Entropía Cruzada)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(historial.history['accuracy'], label='Precisión (Entrenamiento)')
plt.plot(historial.history['val_accuracy'], label='Precisión (Validación)')
plt.title('Curva de Precisión (Accuracy)')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.tight_layout()
plt.show() # Guarda esta imagen para el Word

# 2. Generar predicciones para la Matriz y el Reporte
y_pred_probs = modelo.predict(X_test)
y_pred_clases = np.argmax(y_pred_probs, axis=1)
y_test_clases = np.argmax(y_test, axis=1)

# Imprimir Reporte de Clasificación (F1-Score, Precision, Recall)
print("\n--- REPORTE DE CLASIFICACIÓN DETALLADO ---")
print(classification_report(y_test_clases, y_pred_clases, target_names=["Nivel 1", "Nivel 2", "Nivel 3", "Nivel 4", "Nivel 5"]))

# 3. Dibujar Matriz de Confusión
cm = confusion_matrix(y_test_clases, y_pred_clases)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["N1", "N2", "N3", "N4", "N5"], yticklabels=["N1", "N2", "N3", "N4", "N5"])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción de la Red Neuronal')
plt.ylabel('Nivel de Riesgo Real')
plt.show() # Guarda esta imagen para el Word