import json
from keras.models import Sequential
from keras.layers import Dense, Input
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf
import random
import time
import psutil

# Establecer semillas para reproducibilidad
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Variable booleana para controlar si se debe calcular o cargar los hiperparámetros
calcular_hiperparametros = False
param_file = 'algorythms/ann/best_hyper_ANN.json'

# Cargar los datos
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Codificar las etiquetas a valores numéricos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# Crear el modelo (esto es necesario para usar KerasClassifier)
def create_model(optimizer='adam', kernel_initializer='uniform'):
    model = Sequential()
    model.add(Input(shape=(data.shape[1],)))
    model.add(Dense(64, kernel_initializer=kernel_initializer, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Crear el modelo KerasClassifier usando Sci-Keras
model = KerasClassifier(model=create_model, verbose=0)

# Ajuste de hiperparámetros mediante GridSearchCV y validación cruzada
if calcular_hiperparametros:
    param_grid = {
        'batch_size': [10, 32, 64],
        'epochs': [75, 100, 125, 150],
        'model__optimizer': ['adam', 'rmsprop', 'sgd'],
        'model__kernel_initializer': ['uniform', 'glorot_uniform', 'he_uniform']
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=2)
    grid_result = grid.fit(data, y_encoded)

    print(f"Mejores hiperparámetros: {grid_result.best_params_}")
    print(f"Mejor precisión: {grid_result.best_score_}")

    with open(param_file, 'w') as f:
        json.dump(grid_result.best_params_, f)

    best_params = grid_result.best_params_

else:
    with open(param_file, 'r') as f:
        best_params = json.load(f)

    print(f"Mejores hiperparámetros cargados: {best_params}")

# Crear y entrenar el modelo con los mejores hiperparámetros
print("\n--- Entrenando el modelo real ---")
best_model = create_model(
    optimizer=best_params['model__optimizer'],
    kernel_initializer=best_params['model__kernel_initializer']
)

best_model.fit(
    data, y_encoded,
    batch_size=best_params['batch_size'],
    epochs=best_params['epochs'],
    verbose=0
)

# Evaluar el modelo en los datos completos
loss, accuracy = best_model.evaluate(data, y_encoded, verbose=0)
print(f'Acierto en el conjunto completo: {accuracy * 100:.2f}%')

# -------------------------------
# Medición de recursos
# -------------------------------
def medir_recursos(func):
    def wrapper(*args, **kwargs):
        proceso = psutil.Process(os.getpid())
        
        # Inicializar las métricas de CPU y tiempo
        proceso.cpu_percent(interval=None)
        inicio_tiempo = time.time()
        
        resultado = func(*args, **kwargs)
        time.sleep(0.1) 
        
        # Calcular las métricas finales
        fin_cpu = proceso.cpu_percent(interval=None)
        fin_tiempo = time.time()
        
        uso_cpu = fin_cpu
        tiempo_ejecucion = fin_tiempo - inicio_tiempo
        
        print(f"Tiempo de ejecución: {tiempo_ejecucion:.2f} segundos")
        print(f"CPU usada: {uso_cpu:.2f}%")
        
        return resultado
    return wrapper

# -------------------------------
# Función de Validación Cruzada
# -------------------------------
@medir_recursos
def cross_validation(k, mejores_hiperparametros, x, y):
    model_cv = KerasClassifier(
        model=create_model,
        optimizer=mejores_hiperparametros['model__optimizer'],
        kernel_initializer=mejores_hiperparametros['model__kernel_initializer'],
        batch_size=mejores_hiperparametros['batch_size'],
        epochs=mejores_hiperparametros['epochs'],
        verbose=0,
        random_state=seed
    )

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    print("\n--- Ejecutando Validación Cruzada ---")
    scores = cross_val_score(
        model_cv,
        x,
        y,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1
    )

    std_score = np.std(scores)
    sem_score = std_score / np.sqrt(k)

    print(f'Precisión media con {k}-Fold Cross-Validation: {scores.mean() * 100:.2f}%')
    print(f'Desviación estándar: {std_score * 100:.2f}%')
    print(f'Error estándar de la media: {sem_score * 100:.2f}%')

    return scores.mean(), scores.std()

# Ejecutar Validación Cruzada
media_accuracy, std_accuracy = cross_validation(9, best_params, data, y_encoded)
