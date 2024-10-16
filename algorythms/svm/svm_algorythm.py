import os
import pickle
import json
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
import seaborn as sns

# Directorio base para resultados
results_dir = 'algorythms/svm/results'

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
repeat = True

script_dir = os.path.dirname(os.path.realpath(__file__))
param_file = os.path.join(script_dir, 'best_hyper_svm.json')

# Ajuste de hiperparámetros si calcular_hiperparametros es True
if calcular_hiperparametros:
    param_grid = {
        'C': [0.001,0.01,0.1,1,10,100,1000],
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
def cross_validation(model, data, labels):
    k = 9  # Número de pliegues
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Evaluar el modelo usando validación cruzada
    scores = cross_val_score(model, data, labels, cv=skf, scoring='accuracy', n_jobs=-1)

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    sem_score = std_score / np.sqrt(k)
    print(f'El acierto medio con {k}-Fold Cross-Validation: {mean_score * 100:.2f}%')
    print(f'Desviación estándar: {std_score * 100:.2f}%')
    print(f'El error estándar de la media es {sem_score * 100:.2f}%')

    # Generar predicciones y matriz de confusión
    y_pred = cross_val_predict(model, data, labels, cv=skf)
    class_report = classification_report(labels, y_pred)
    conf_matrix = confusion_matrix(labels, y_pred)

    # Guardar el informe de clasificación
    report_path =  os.path.join(results_dir, 'classification_report_cv.txt')
    with open(report_path, 'w') as f:
        f.write(class_report)

    # Graficar la matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(labels),
                yticklabels=np.unique(labels))
    plt.title(f'Matriz de confusión - SVM con {k}-Fold Cross-Validation')
    plt.xlabel('Etiqueta Predicha')
    plt.ylabel('Etiqueta Verdadera')
    cm_path = os.path.join(results_dir, f'confusion_matrix_svm_{k}_fold.png')
    plt.savefig(cm_path)
    plt.close()

    return scores.mean(), scores.std()

# -----------------------------------------------------
# Alternativa: Repetición de Entrenamiento
# -----------------------------------------------------
@medir_recursos
def repeat_training(n_it, x_train, y_train, x_test, y_test):
    
    scores = []
    cumulative_mean = []
    cumulative_std = []

    for i in range(n_it):
        x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=None
        )

        model_r = SVC(**mejores_hiperparametros, random_state=None)

        #Entrenar el modelo
        model_r.fit(x_train, y_train)
        y_predict = model_r.predict(x_test)
        
        #Calcular métricas
        score = accuracy_score(y_predict, y_test)
        scores.append(score)
        current_mean = np.mean(scores)
        current_std = np.std(scores)
        cumulative_mean.append(current_mean)
        cumulative_std.append(current_std)

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    sem_score = std_score / np.sqrt(n_it)
    print(f'El acierto medio después de {n_it} iteraciones es de: {mean_score * 100:.2f}%')
    print(f'Desviación estándar: {std_score * 100:.2f}%')
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
    rt_path = os.path.join(results_dir, 'resultados_repeticion_svm.png')
    plt.savefig(rt_path)
    plt.close() 

    return mean_score

# --------------------------------------------------
# Ejecución de Alternativas con Medición de Recursos
# --------------------------------------------------
if cross_val:
    print("\n--- Ejecutando Validación Cruzada ---")
    cross_validation(model, data, labels)

if repeat:
    print("\n--- Ejecutando Repetición de Entrenamiento ---")
    n_it = 30
    repeat_training(n_it, x_train, y_train, x_test, y_test)
