import pickle
import json
import os
import time
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np

# -----------------------------------------------------
# Parámetros para las funciones
# -----------------------------------------------------
calcular_hiperparametros = False

cross_val = False
cv_folds = 9 # Nº de pliegues validación cruzada

adjust_cross_val = False
k_values=[5,6,7,8,9,10,11,12,13,14,15,16,17,18] # Valores para el ajuste de validación cruzada

repeat = True
n_it = 30 # Nº de iteraciones de la repetición de entrenamiento

# Ruta del archivo JSON para guardar/cargar los mejores hiperparámetros
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, 'best_hyper_rf.json')

# Cargar los datos
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])


# -------------------------------
# Ajuste de hiperparámeros
# -------------------------------
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
    model = RandomForestClassifier(random_state=42)

    # Configurar el GridSearchCV para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

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
        mejores_hiperparámetros = json.load(f)

    print(f"Hiperparámetros cargados: {mejores_hiperparámetros}")

# Definir el modelo utilizando los hiperparámetros cargados o encontrados
'''model = RandomForestClassifier(
    n_estimators=mejores_hiperparámetros['n_estimators'],
    max_depth=mejores_hiperparámetros['max_depth'],
    bootstrap=mejores_hiperparámetros['bootstrap'],
    min_samples_leaf=mejores_hiperparámetros['min_samples_leaf'],
    min_samples_split=mejores_hiperparámetros['min_samples_split']
)'''

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
# Alternativa: Validación Cruzada
# -------------------------------
@medir_recursos
def cross_validation(k):

    model = RandomForestClassifier(
    n_estimators=mejores_hiperparámetros['n_estimators'],
    max_depth=mejores_hiperparámetros['max_depth'],
    bootstrap=mejores_hiperparámetros['bootstrap'],
    min_samples_leaf=mejores_hiperparámetros['min_samples_leaf'],
    min_samples_split=mejores_hiperparámetros['min_samples_split'],
    random_state=None
    )

    scores = cross_val_score(model, data, labels, cv=k, scoring='accuracy', n_jobs=-1)

    # Mostrar la precisión media y la desviación estándar
    print(f'Precisión media con {k}-Fold Cross-Validation: {scores.mean() * 100:.2f}%')
    print(f'Desviación estándar: {scores.std() * 100:.2f}%')
    
    return scores.mean(), scores.std()


# -----------------------------------------------------
# Alternativa: Repetición de Entrenamiento
# -----------------------------------------------------
@medir_recursos
def repeat_training(n_it):

    scores = []

    for i in range(n_it):

        x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=None)

        model = RandomForestClassifier(
        n_estimators=mejores_hiperparámetros['n_estimators'],
        max_depth=mejores_hiperparámetros['max_depth'],
        bootstrap=mejores_hiperparámetros['bootstrap'],
        min_samples_leaf=mejores_hiperparámetros['min_samples_leaf'],
        min_samples_split=mejores_hiperparámetros['min_samples_split'],
        random_state=None)

        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        
        score = accuracy_score(y_predict, y_test)
        scores.append(score)
        
        print(f'{score * 100:.2f}% de las muestras fueron clasificadas correctamente en la iteración {i+1}!')

    # Calcula la media de los scores
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f'Average score after {n_it} iterations: {mean_score * 100:.2f}% ± {std_score * 100:.2f}%')


    return mean_score


# -----------------------------------------------------
# Llamadas a las funciones
# -----------------------------------------------------
if repeat:
    print("\n--- Ejecutando Repetición de Entrenamiento ---")
    repeat_training(n_it)
    
if cross_val:
    print("\n--- Ejecutando Validación Cruzada ---")
    cross_validation(cv_folds)

if adjust_cross_val:
    print("\n--- Ejecutando el ajuste de Validación Cruzada ---")

    for k in k_values:
        cross_validation(k)
        print("\n")
