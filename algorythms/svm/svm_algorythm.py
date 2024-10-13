import os
import pickle
import json
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
import seaborn as sns

# Cargar los datos
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Semilla aleatoria
random_state = 42

# Dividir los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=random_state
)

# Escalar los datos
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

calcular_hiperparametros = False
cross_val = True
repeat = False

script_dir = os.path.dirname(os.path.realpath(__file__))
param_file = os.path.join(script_dir, 'best_hyper_svm.json')

# Ajuste de hiperparámetros si calcular_hiperparametros es True
if calcular_hiperparametros:
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    model = SVC()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    mejores_hiperparametros = grid_search.best_params_
    with open(param_file, 'w') as f:
        json.dump(mejores_hiperparametros, f)
    print(f"Mejores hiperparámetros encontrados: {mejores_hiperparametros}")
else:
    with open(param_file, 'r') as f:
        mejores_hiperparametros = json.load(f)
    print(f"Hiperparámetros cargados: {mejores_hiperparametros}")

# Crear el modelo con los mejores hiperparámetros
model = SVC(**mejores_hiperparametros, random_state=random_state)

# -------------------------------
# Medición de recursos
# -------------------------------
def medir_recursos(func):
    def wrapper(*args, **kwargs):
        proceso = psutil.Process(os.getpid())
        proceso.cpu_percent(interval=None)
        inicio_tiempo = time.time()
        
        resultado = func(*args, **kwargs)
        time.sleep(0.1)
        
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
def cross_validation():
    k = 9  # Número de pliegues
    scores = cross_val_score(model, data, labels, cv=k, scoring='accuracy', n_jobs=-1)
    print(f'Precisión media con {k}-Fold Cross-Validation: {scores.mean() * 100:.2f}%')
    print(f'Desviación estándar: {scores.std() * 100:.2f}%')

    # Generar predicciones y matriz de confusión
    y_pred = cross_val_predict(model, data, labels, cv=k)
    class_report = classification_report(labels, y_pred)
    conf_matrix = confusion_matrix(labels, y_pred)

    # Guardar el informe de clasificación
    report_path = 'algorythms/svm/results/classification_report_cv.txt'
    with open(report_path, 'w') as f:
        f.write(class_report)

    # Graficar la matriz de confusión
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de confusión - Validación Cruzada')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('algorythms/svm/results/confusion_matrix_cv.png')
    plt.close()

    return scores.mean(), scores.std()

# -----------------------------------------------------
# Alternativa: Repetición de Entrenamiento
# -----------------------------------------------------
@medir_recursos
def repeat_training(n_it, model, x_train, y_train, x_test, y_test):
    scores = []
    for i in range(n_it):
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        score = accuracy_score(y_predict, y_test)
        scores.append(score)

    mean_score = sum(scores) / n_it
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
