import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Configurar pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None)

# Cargar el conjunto de datos de precios de casas de California
housing = fetch_california_housing()

columnas = ['IngresoMediano', 'EdadCasa', 'PromHabitaciones', 'PromDormitorios',
            'Población', 'PromOcupación', 'Latitud', 'Longitud']

# (IngresoMediano): Mediana del ingreso por hogares en el bloque (en decenas de miles de dólares *10k ).
# (EdadCasa): Edad mediana de las casas en el bloque (en años).
# (PromHabitaciones): Promedio de habitaciones por casa.
# (PromDormitorios): Promedio de dormitorios por casa.
# (Población): Población del bloque.
# (PromOcupación): Promedio de ocupación (número de residentes por casa).
# (Latitud): Latitud del bloque.
# (Longitud): Longitud del bloque.
# (Precio): Valor medio de las casas en el bloque (en decenas de miles de dólares *10k).

# Crear el DataFrame con las 8 columnas
data = pd.DataFrame(housing.data, columns=columnas)

# Agregar la columna 'Precio' utilizando los valores de housing.target
data['Precio'] = housing.target

# Limitar el conjunto de datos a los primeros 10,000 registros
#data = data.iloc[:10000]

# Mostrar un resumen de los datos
print("Data info")
print(data.info())

# Mostrar todos los encabezados
print("Encabezados de las columnas:")
print(data.columns)

# Mostrar las primeras filas del DataFrame
print("\nPrimeros 50 registros del conjunto de datos:")
print(data.head(50))

# Estadísticas descriptivas del DataFrame
print(data.describe())

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop('Precio', axis=1)
y = data['Precio']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)
#20% de los datos se utilizarán como conjunto de prueba, y el 80% restante se utilizarán para el entrenamiento
#None para semilla aleatoria


# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualizar las predicciones frente a los valores reales
plt.figure(figsize=(15, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Precios Reales', alpha=0.6)
plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicciones', alpha=0.6)
plt.plot(range(len(y_test)), y_test, color='blue', alpha=0.3)
plt.plot(range(len(y_test)), y_pred, color='red', alpha=0.3)
plt.xlabel("Índice de Observaciones")
plt.ylabel("Precio de la Vivienda (en decenas de miles de dólares)")
plt.title("Comparación de Predicciones y Precios Reales de Viviendas en California")
plt.legend()
plt.show()
