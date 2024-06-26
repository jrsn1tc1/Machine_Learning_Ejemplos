import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Cargar el conjunto de datos de precios de casas de California
housing = fetch_california_housing()

# Convertir los datos en un DataFrame de pandas
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

# Mostrar un resumen de los datos
print(data.info())

# Mostrar las primeras filas del DataFrame
print(data.head())

# Estadísticas descriptivas del DataFrame
print(data.describe())
