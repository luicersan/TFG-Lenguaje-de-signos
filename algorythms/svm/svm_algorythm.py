import os
import pickle
import json
import numpy as np
import time
import psutil
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

# Escalar los datos
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

calcular_hiperparametros = False  
script_dir = os.path.dirname(os.path.realpath(__file__))
param_file = os.path.join(script_dir, 'best_hyper_svm.json') 

# Ajuste de hiperparámetros si calcular_hiperparametros es True
if calcular_hiperparametros:
    # Definir los hiperparámetros a ajustar
    param_grid = {
        'C': [0.1, 1, 10, 100], 
        'kernel': ['linear', 'rbf', 'poly'], 
        'gamma': ['scale', 'auto']
    }

    # Crear el modelo SVM
    model = SVC()

    # Configurar el GridSearchCV para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Ajustar el modelo usando la búsqueda de hiperparámetros
    grid_search.fit(x_train, y_train)

    # Obtener los mejores hiperparámetros
    mejores_hiperparametros = grid_search.best_params_

    # Guardar los mejores hiperparámetros en un archivo JSON
    with open(param_file, 'w') as f:
        json.dump(mejores_hiperparametros, f)

    print(f"Mejores hiperparámetros encontrados: {mejores_hiperparametros}")

else:
    # Cargar los mejores hiperparámetros desde el archivo JSON
    with open(param_file, 'r') as f:
        mejores_hiperparametros = json.load(f)

    print(f"Hiperparámetros cargados: {mejores_hiperparametros}")

# Crear el modelo con los mejores hiperparámetros
model = SVC(**mejores_hiperparametros)
cross_val = True
repeat = True

# -------------------------------
# Medición de recursos
# -------------------------------
def medir_recursos(func):
    def wrapper(*args, **kwargs):
        # Inicializar el uso de CPU y memoria
        proceso = psutil.Process(os.getpid())
        inicio_memoria = proceso.memory_info().rss / (1024 * 1024)  # En MB
        inicio_cpu = proceso.cpu_percent(interval=None)
        inicio_tiempo = time.time()

        resultado = func(*args, **kwargs)

        # Calcular los recursos usados
        fin_memoria = proceso.memory_info().rss / (1024 * 1024)  # En MB
        fin_cpu = proceso.cpu_percent(interval=None)
        fin_tiempo = time.time()

        uso_memoria = fin_memoria - inicio_memoria
        uso_cpu = fin_cpu - inicio_cpu
        tiempo_ejecucion = fin_tiempo - inicio_tiempo

        print(f"Tiempo de ejecución: {tiempo_ejecucion:.2f} segundos")
        print(f"Memoria usada: {uso_memoria:.2f} MB")
        print(f"CPU usada: {uso_cpu:.2f}%")

        return resultado
    return wrapper


# -------------------------------
# Alternativa: Validación Cruzada
# -------------------------------
@medir_recursos
def cross_validation():
    k = 10  # Número de pliegues
    scores = cross_val_score(model, data, labels, cv=k, scoring='accuracy', n_jobs=-1)

    # Mostrar la precisión media y la desviación estándar
    print(f'Precisión media con {k}-Fold Cross-Validation: {scores.mean() * 100:.2f}%')
    print(f'Desviación estándar: {scores.std() * 100:.2f}%')
    return scores.mean(), scores.std()


# -----------------------------------------------------
# Alternativa: Repetición de Entrenamiento
# -----------------------------------------------------
@medir_recursos
def repeat_training(n_it, model, x_train, y_train, x_test, y_test):
    scores = []

    for i in range(n_it):
        # Entrena el modelo
        model.fit(x_train, y_train)
         
        # Realiza predicciones
        y_predict = model.predict(x_test)
            
        # Calcula el score
        score = accuracy_score(y_predict, y_test)
        scores.append(score)

    # Calcula la media de los scores
    mean_score = sum(scores) / n_it
    print('Sin redondear: ', mean_score)
    print('Average score after {} iterations: {}%'.format(n_it, round(mean_score, 4) * 100))

    return mean_score

# --------------------------------------------------
# Ejecución de Alternativas con Medición de Recursos
# --------------------------------------------------
if cross_val:
    print("\n--- Ejecutando Validación Cruzada ---")
    cross_validation()

if repeat:
    print("\n--- Ejecutando Repetición de Entrenamiento ---")
    n_it = 50
    repeat_training(n_it, model, x_train, y_train, x_test, y_test)
