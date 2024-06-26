import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None)

# Cargar el conjunto de datos de diabetes
diabetes = load_diabetes()

# Nombres de las columnas en español
columnas = ['Edad', 'Sexo', 'IMC', 'Presión Arterial', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']

# Crear el DataFrame con las características
data = pd.DataFrame(diabetes.data, columns=columnas)

# Agregar la columna 'Progresión' utilizando los valores de diabetes.target
data['Progresión'] = diabetes.target

# Mostrar un resumen de los datos
print("Información del conjunto de datos:")
print(data.info())

# Mostrar todos los encabezados
print("\nEncabezados de las columnas:")
print(data.columns)

# Mostrar las primeras 50 filas del DataFrame
print("\nPrimeros 50 registros del conjunto de datos:")
print(data.head(50))

# Estadísticas descriptivas del DataFrame
print("\nEstadísticas descriptivas del conjunto de datos:")
print(data.describe())

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop('Progresión', axis=1)
y = data['Progresión']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Visualizar las predicciones frente a los valores reales con un gráfico de dispersión
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicciones')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Línea Ideal')
plt.xlabel("Progresión Real")
plt.ylabel("Progresión Predicha")
plt.title("Progresión Real vs Predicha de Diabetes")
plt.legend()
plt.show()

# Visualizar la distribución de errores
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True, color='purple')
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.title("Distribución de Errores")
plt.legend(['Errores', 'Densidad'])
plt.show()

# Visualizar la importancia de las características
plt.figure(figsize=(10, 6))
coef = pd.Series(model.coef_, index=X.columns)
coef.sort_values().plot(kind='barh', color='green')
plt.xlabel("Coeficiente")
plt.title("Importancia de las Características")
plt.show()
