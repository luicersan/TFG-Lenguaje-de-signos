import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Soluciones de MP para trazar los landmarks encima de las imágenes
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = [] #Nombre de las carpetas de las imágenes
# Recorremos cada uno de los archivos dentro de las carpetas de data
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # Convertimos la imagen en RGB para que pueda ser leída por MP
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        # Iteramos sobre los landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    # Guardamos las coordenadas x e y de los landmarks para entrenar posteriormente el modelo
                    #print(hand_landmarks.landmark[i])
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalización de coordenadas para que los mínimos valores de x e y sean (0,0)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux) #Agregamos el array de coordenadas normalizadas
            labels.append(dir_) #Nombre de las carpetas de las imágenes 

# Guardamos el diccionario con datos y etiquetas
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f) 
f.close()
