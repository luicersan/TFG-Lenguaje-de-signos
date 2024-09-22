# Uso del Machine Learning para el reconocimiento del alfabeto dactilológico español

## Introducción

Este proyecto tiene como objetivo principal la clasificación de imágenes de manos representando diferentes letras del **alfabeto dactilológico español** (lengua de señas). Las imágenes son capturadas y procesadas para su clasificación a través de varios algoritmos de Machine Learning, y se comparan dichos algoritmos en términos de precisión, rendimiento y sostenibilidad.

## Objetivos del Proyecto

- Desarrollar un sistema capaz de identificar gestos manuales que representan letras del alfabeto dactilológico español.
- Implementar y comparar distintos algoritmos de clasificación en cuanto a precisión, velocidad de procesamiento y eficiencia computacional.

## Descripción del Proceso

1. **Captura de imágenes**: Las imágenes de las manos se capturan utilizando un script automatizado.
2. **Extracción de características**: A través de la librería **MediaPipe** de Google, las imágenes son transformadas en un conjunto de coordenadas que describen los puntos clave de las manos.
3. **Clasificación**: Los datos extraídos son alimentados a varios algoritmos de Machine Learning para su clasificación en diferentes letras del alfabeto.
4. **Comparativa de Algoritmos**: Los algoritmos serán evaluados y comparados en base a:
   - Precisión en la clasificación.
   - Rendimiento computacional (tiempo de procesamiento, uso de recursos).
   - Sostenibilidad (consumo de energía o eficiencia de recursos).

## Algoritmos Utilizados

- **Algoritmo 1**: Random Forests, o bosques aleatorios.
- **Algoritmo 2**: Support vector machine.
- **Algoritmo 3**: Red neuronal artificial.

## Requisitos principales del Proyecto

Puedes encontrar todas las dependencias del proyecto en el archivo requirements.txt.

## Ejecución

Para ejecutar el proyecto, sigue los siguientes pasos:
## 1. Instalar las dependencias del proyecto, se recomienda hacerlo en un entorno virtual:
#### Crear entorno virtual
python -m venv nombre_entorno

#### Activar entorno virtual
#### - En Windows:
nombre_entorno\Scripts\activate
#### - En macOS/Linux:
source nombre_entorno/bin/activate

#### Instalar dependencias
pip install -r requirements.txt

#### Desactivar entorno virtual
deactivate

## 2. Ejecutar el script de captura de imágenes 'collect_imgs.py'.
Este script abre la cámara del dispositivo y captura imágenes ininterrumpidamente, clasificándolas en subcarpetas según los parámetros de entrada. Este script dispone de los siguientes parámetros de entrada, a configurar por el usuario:
- Especificar en la variable DATA_DIR el nombre de la carpeta principal que contendrá todas las subcarpetas.
- Especificar en la variable letters_array las letras a capturar, las letras especificadas en esta variable serán los nombres de las subcaretas que contengan las imágenes.
- Especificar en la variable dataset_size el número de imágenes a capturar de cada letra.
- Indicar en la variable wait_frames el tiempo en milisegundos entre captura y captura.

Por ejemplo:

DATA_DIR = './data'

letters_array = ['a','b','c]

dataset_size = 10

wait_frames = 1200

## 3. Crear el dataset a partir de las imágenes capturadas 'create_dataset.py'.
Gracias a la librería MediaPipe de Google, las imágenes capturadas se convierten en unos arrays de coordenadas de los puntos de interés de cada mano, de modo que los algoritmos puedan reconocer estas imágenes.

Estas coordenadas se guardan en un archivo data.pickle, que será el que le pasemos como entrada a los distintos algoritmos.

Para ejecutar este script basta con indicar la ruta de la carpeta data, que contiene las subcarpetas con las diferentes imágenes capturadas y clasificadas anteriormente.


## 3. Ejecución de los algoritmos de procesamiento y clasificación.
Dentro de la carpeta "algorythms" se encuentran varias subcarpetas con cada uno de los algoritmos implementados. Cada uno de los algoritmos cuenta con una opción de ajustar sus hiperparámetros, que solo es necesaria activar cuando cambie el dataset, permitiendo así ahorrar recursos.

## Conclusión y Trabajos Futuros

Este proyecto no solo busca desarrollar un clasificador de imágenes eficiente, sino también analizar y comparar distintos enfoques para encontrar el más adecuado en términos de precisión y sostenibilidad. En el futuro, se podría ampliar el proyecto para incluir gestos dinámicos, incluir un conjunto de palabras enteras, o implementar el procesamiento de imágenes en tiempo real.
