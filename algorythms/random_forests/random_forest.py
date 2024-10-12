import pickle
import json
import os
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import random

# -----------------------------------------------------
# 1. Configuración de Semillas Aleatorias para Reproducibilidad
# -----------------------------------------------------
seed = 42
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# -----------------------------------------------------
# 2. Definición de Directorios de Salida
# -----------------------------------------------------
# Directorio base para resultados
output_dir = 'algorythms/random_forests/results_rf'

# Subdirectorios para diferentes tipos de resultados
confusion_matrix_dir = os.path.join(output_dir, 'confusion_matrices')
classification_report_dir = os.path.join(output_dir, 'classification_reports')

# Crear los directorios si no existen
os.makedirs(confusion_matrix_dir, exist_ok=True)
os.makedirs(classification_report_dir, exist_ok=True)

# -----------------------------------------------------
# 3. Parámetros para las funciones
# -----------------------------------------------------
calcular_hiperparametros = False

cross_val = True
cv_folds = 9  # Nº de pliegues validación cruzada

adjust_cross_val = False
k_values = [5,6,7,8,9,10,11,12,13,14,15,16,17,18]  # Valores para el ajuste de validación cruzada

repeat = False
n_it = 30  # Nº de iteraciones de la repetición de entrenamiento

# Ruta del archivo JSON para guardar/cargar los mejores hiperparámetros
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, 'best_hyper_rf.json')

# -----------------------------------------------------
# 4. Cargar los datos
# -----------------------------------------------------
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# -----------------------------------------------------
# 5. Ajuste de hiperparámetros
# -----------------------------------------------------
if calcular_hiperparametros:
    # Definir los hiperparámetros a ajustar
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Crear el modelo RandomForestClassifier
    model = RandomForestClassifier(random_state=seed)

    # Configurar el GridSearchCV para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

    # Ajustar el modelo usando la búsqueda de hiperparámetros
    grid_search.fit(data, labels)

    # Obtener los mejores hiperparámetros
    mejores_hiperparametros = grid_search.best_params_

    # Guardar los mejores hiperparámetros en un archivo JSON
    with open(json_path, 'w') as f:
        json.dump(mejores_hiperparametros, f)

    print(f"Mejores hiperparámetros encontrados: {mejores_hiperparametros}")

else:
    # Cargar los mejores hiperparámetros desde el archivo JSON
    with open(json_path, 'r') as f:
        mejores_hiperparametros = json.load(f)

    print(f"Hiperparámetros cargados: {mejores_hiperparametros}")

# -----------------------------------------------------
# 6. Medición de recursos
# -----------------------------------------------------
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
        
        # Retornar los resultados junto con uso_cpu y tiempo_ejecucion
        return resultado[0], resultado[1], uso_cpu, tiempo_ejecucion
    return wrapper

# -----------------------------------------------------
# 7. Función de Validación Cruzada
# -----------------------------------------------------
@medir_recursos
def cross_validation(k, mejores_hiperparametros, x, y):
    model_cv = RandomForestClassifier(
        n_estimators=mejores_hiperparametros['n_estimators'],
        max_depth=mejores_hiperparametros['max_depth'],
        bootstrap=mejores_hiperparametros['bootstrap'],
        min_samples_leaf=mejores_hiperparametros['min_samples_leaf'],
        min_samples_split=mejores_hiperparametros['min_samples_split'],
        random_state=seed  # Asegurar reproducibilidad
    )

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    
    # Realizar la validación cruzada
    scores = cross_val_score(
        model_cv,
        x,
        y,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1
    )

    # Obtener predicciones para matriz de confusión y reporte de clasificación
    y_pred = cross_val_predict(
        model_cv,
        x,
        y,
        cv=skf,
        n_jobs=-1
    )

    # Generar y guardar la matriz de confusión
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y),
                yticklabels=np.unique(y))
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.title(f'Matriz de Confusión - {k}-Fold Cross-Validation')
    cm_path = os.path.join(confusion_matrix_dir, f'confusion_matrix_{k}_fold.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Matriz de confusión guardada en: {cm_path}")

    # Generar y guardar el informe de clasificación
    report = classification_report(y, y_pred, target_names=np.unique(y).astype(str))
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

# -----------------------------------------------------
# 8. Alternativa: Repetición de Entrenamiento
# -----------------------------------------------------
@medir_recursos
def repeat_training(n_it, labels, test_size):
    """
    Repite el entrenamiento del modelo varias veces con diferentes divisiones de datos y aleatoriedad en el modelo, promediando los resultados.

    :param n_it: Número de iteraciones
    :param labels: Etiquetas
    :param test_size: Proporción del conjunto de prueba
    :return: Media y desviación estándar de las puntuaciones de precisión
    """
    scores = []
    cumulative_mean = []
    cumulative_std = []
    
    for i in range(n_it):
        # Dividir los datos de manera aleatoria en cada iteración
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=test_size, stratify=labels, random_state=None, shuffle=True
        )

        model = RandomForestClassifier(
            n_estimators=mejores_hiperparametros['n_estimators'],
            max_depth=mejores_hiperparametros['max_depth'],
            bootstrap=mejores_hiperparametros['bootstrap'],
            min_samples_leaf=mejores_hiperparametros['min_samples_leaf'],
            min_samples_split=mejores_hiperparametros['min_samples_split'],
            random_state=None
        )

        # Entrenar el modelo
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)

        # Calcular métricas
        score = accuracy_score(y_test, y_predict)
        scores.append(score)
        current_mean = np.mean(scores)
        current_std = np.std(scores)
        cumulative_mean.append(current_mean)
        cumulative_std.append(current_std)

    # Calcular la media y desviación de los scores
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    sem_score = std_score / np.sqrt(n_it)
    print(f'El acierto medio después de {n_it} iteraciones es de: {mean_score * 100:.2f}% ± {std_score * 100:.2f}% (SEM: {sem_score * 100:.2f}%)')
    print(f'El error estándar de la media es {sem_score * 100:.2f}%')

    # Visualizar los resultados
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, n_it + 1), scores, marker='o', linestyle='-', color='blue', label='Precisión por Iteración')
    plt.plot(range(1, n_it + 1), cumulative_mean, marker='x', linestyle='--', color='red', label='Media Acumulada')
    plt.plot(range(1, n_it + 1), [m + s for m, s in zip(cumulative_mean, cumulative_std)], 
             linestyle=':', color='green', label='Media + Desviación Estándar')
    plt.plot(range(1, n_it + 1), [m - s for m, s in zip(cumulative_mean, cumulative_std)], 
             linestyle=':', color='green', label='Media - Desviación Estándar')
    plt.title('Resultados de Repetición de Entrenamiento')
    plt.xlabel('Iteración')
    plt.ylabel('Precisión (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('algorythms/random_forests/resultados_repeticion_entrenamiento.png')
    plt.close() 

    return mean_score, std_score, sem_score

# -----------------------------------------------------
# 9. Llamadas a las funciones
# -----------------------------------------------------
if repeat:
    print("\n--- Ejecutando Repetición de Entrenamiento ---")
    mean, std, sem = repeat_training(n_it, labels, test_size=0.2)
    
if cross_val:
    print("\n--- Ejecutando Validación Cruzada ---")
    media_accuracy, std_accuracy, uso_cpu, tiempo_ejecucion = cross_validation(cv_folds, mejores_hiperparametros, data, labels)
    
if adjust_cross_val:
    print("\n--- Ejecutando el ajuste de Validación Cruzada ---")
    for k in k_values:
        media_accuracy, std_accuracy, uso_cpu, tiempo_ejecucion = cross_validation(k, mejores_hiperparametros, data, labels)
        print("\n")
