import os
import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_letters = 3
letters_array = ['a','b','c']
dataset_size = 10
wait_frames = 100

cap = cv2.VideoCapture(0)
for j in range(len(letters_array)):
    print(os.path.join(DATA_DIR, letters_array[j]))
    if not os.path.exists(os.path.join(DATA_DIR, letters_array[j])):
        os.makedirs(os.path.join(DATA_DIR, letters_array[j]))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, '¿Preparado? Presione "Q" para empezar a capturar :)', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(wait_frames)
        cv2.imwrite(os.path.join(DATA_DIR, letters_array[j], '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
