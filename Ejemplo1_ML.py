import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Cargar el conjunto de datos de precios de casas de California
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop('PRICE', axis=1)
y = data['PRICE']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualizar las predicciones frente a los valores reales
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Precios Reales', alpha=0.6)
plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicciones', alpha=0.6)
plt.xlabel("Índice")
plt.ylabel("Precio de la Vivienda")
plt.title("Comparación de Predicciones y Precios Reales")
plt.legend()
plt.show()
