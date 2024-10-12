import json
from keras.models import Sequential
from keras.layers import Dense, Input
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf
import random
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# -------------------------------
# 1. Definición de Directorios de Salida
# -------------------------------
# Directorio base para resultados
output_dir = 'algorythms/ann/results'

# Subdirectorios para diferentes tipos de resultados
confusion_matrix_dir = os.path.join(output_dir, 'confusion_matrices')
classification_report_dir = os.path.join(output_dir, 'classification_reports')

# Crear los directorios si no existen
os.makedirs(confusion_matrix_dir, exist_ok=True)
os.makedirs(classification_report_dir, exist_ok=True)

# -------------------------------
# 2. Establecer semillas para reproducibilidad
# -------------------------------
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# -------------------------------
# 3. Variables de Control
# -------------------------------
calcular_hiperparametros = False
param_file = 'algorythms/ann/best_hyper_ANN.json'

# -------------------------------
# 4. Cargar los datos
# -------------------------------
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Codificar las etiquetas a valores numéricos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# -------------------------------
# 5. Crear el modelo (esto es necesario para usar KerasClassifier)
# -------------------------------
def create_model(optimizer='adam', kernel_initializer='uniform'):
    model = Sequential()
    model.add(Input(shape=(data.shape[1],)))
    model.add(Dense(64, kernel_initializer=kernel_initializer, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------
# 6. Crear el modelo KerasClassifier usando Sci-Keras
# -------------------------------
model = KerasClassifier(model=create_model, verbose=0)

# -------------------------------
# 7. Ajuste de hiperparámetros mediante GridSearchCV y validación cruzada
# -------------------------------
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

# -------------------------------
# 8. Crear y entrenar el modelo con los mejores hiperparámetros
# -------------------------------
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
# 9. Medición de recursos
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
        
        # Desempaquetar el resultado y retornar todos los valores en una tupla plana
        return resultado[0], resultado[1], uso_cpu, tiempo_ejecucion
    return wrapper

# -------------------------------
# 10. Función de Validación Cruzada
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
    scores = cross_val_score(
        model_cv,
        x,
        y,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1
    )

    # Obtener predicciones para matriz de confusión e informe de clasificación
    y_pred = cross_val_predict(
        model_cv,
        x,
        y,
        cv=skf,
        n_jobs=-1
    )

    # Generar y guardar la matriz de confusión
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.title(f'Matriz de Confusión - {k}-Fold Cross-Validation')
    cm_path = os.path.join(confusion_matrix_dir, f'confusion_matrix_{k}_fold.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Matriz de confusión guardada en: {cm_path}")

    # Generar y guardar el informe de clasificación
    report = classification_report(y, y_pred, target_names=label_encoder.classes_)
    report_path = os.path.join(classification_report_dir, f'classification_report_{k}_fold.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Informe de clasificación guardado en: {report_path}")

    std_score = np.std(scores)
    sem_score = std_score / np.sqrt(k)

    print(f'Precisión media con {k}-Fold Cross-Validation: {scores.mean() * 100:.2f}%')
    print(f'Desviación estándar: {std_score * 100:.2f}%')
    print(f'Error estándar de la media: {sem_score * 100:.2f}%')

    return scores.mean(), scores.std()

# -------------------------------
# 11. Llamadas a las Funciones
# -------------------------------
print("\n--- Ejecutando Validación Cruzada ---")
media_accuracy, std_accuracy, uso_cpu, tiempo_ejecucion = cross_validation(9, best_params, data, y_encoded)


# Para comprobar que las etiquetas están correctas
'''class_counts = Counter(labels)
print(class_counts)'''


# -------------------------------
# 12. Evaluación Final del Modelo (Opcional)
# -------------------------------
# Si deseas evaluar el modelo final en el conjunto completo y generar matriz de confusión y reporte
# Puedes descomentar y adaptar el siguiente bloque

# """
# y_pred_final = best_model.predict(data)
# cm_final = confusion_matrix(y_encoded, y_pred_final)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues',
#             xticklabels=label_encoder.classes_,
#             yticklabels=label_encoder.classes_)
# plt.ylabel('Etiqueta Verdadera')
# plt.xlabel('Etiqueta Predicha')
# plt.title('Matriz de Confusión - Modelo Entrenado en Conjunto Completo')
# plt.savefig(os.path.join(confusion_matrix_dir, 'confusion_matrix_final.png'))
# plt.close()

# report_final = classification_report(y_encoded, y_pred_final, target_names=label_encoder.classes_)
# with open(os.path.join(classification_report_dir, 'classification_report_final.txt'), 'w') as f:
#     f.write(report_final)
# print("Evaluación final del modelo completada y guardada.")
# """
