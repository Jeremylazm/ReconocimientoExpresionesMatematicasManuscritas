# Importar las librerias necesarias para usar el modelo CNN
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Importar la libreria pickle para leer archivos pickle
import pickle

# Importar la libreria de sklearn para separar los datos
from sklearn.model_selection import train_test_split

# Importar la libreria os para manejar archivos
import os

# Definir una funcion para cargar los datos y separarlos para train y test
def loadData(fileName,size=0.2):
    # Leer y cargar los datos del archivo .pickle
    with open(fileName, 'rb') as f:
        X, Y = pickle.load(f)

    # Redimensionar los datos de entrada a 45 x 45
    X = X.reshape(-1,45,45,1)

    # Separar los datos para training y testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = size)

    # Retornar los datos separados en train y test
    return X_train, X_test, y_train, y_test

# Definir una funcion para crear el modelo CNN dado 
def createModel(input,output):
    # Constuir un modelo secuencial
    model = Sequential()

    # Las imagenes son de 48 X 48
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input)) #46 X 46
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3,3), activation='relu')) #44 X 44
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3,3), activation='relu')) #44 X 44
    model.add(MaxPooling2D())
    model.add(Dropout(rate=0.15))
    model.add(Flatten()) #1964 X 1
    model.add(Dense(500, activation='relu')) #500 X 1
    model.add(Dropout(0.2))
    model.add(Dense(250, activation='relu')) #250 X 1
    model.add(Dropout(0.2))
    model.add(Dense(125, activation='relu')) #120 X 1
    model.add(Dropout(0.2))
    model.add(Dense(66, activation='softmax')) # 66 X 1 (solo ingles, digitos, and simbolos)
    
    # Compilar el modelo con el optimizador Adam y otros hiperparametros
    model.compile(optimizer = 'adam',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
    
    # Retornar el modelo construido
    return model

# Definir una funcion para cargar el modelo que se usara en el programa
def loadModel(input,output,fileName=None):
    # Verificar si existe el modelo creado
    if fileName is None:
        # Cargar el ultimo modelo
        model = loadLatestModel(input,output)
    else:
        # Crear un modelo
        model = createModel(input,output)
        
        # Cargar los pesos del modelo
        model.load_weights(fileName)

    # Retornar el modelo
    return model

# Definir una funcion para cargar el ultimo modelo   
def loadLatestModel(input,output):
    # Crear un modelo
    model = createModel(input,output) #Currently (45,45,1),65
    
    # Encontrar el nombre del ultimo archivo de punto de control guardado
    latestPath = tf.train.latest_checkpoint('training')

    # Cargar los pesos del modelo
    model.load_weights(latestPath)

    # Retornar el modelo
    return model

# Definir una funcion para entrenar el modelo CNN
def trainModel(model, X_train, y_train, X_test, y_test,ep=50, initial=0):
    # Definir la ruta del punto de control
    checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Devolver la llamada para guardar los pesos del modelo 
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                                     filepath = checkpoint_path,
                                                     verbose = 1,
                                                     save_weights_only=True,
                                                     period = 2
                                                    )
        
    # Guardar los pesos del modelo en un archivo de punto de control
    model.save_weights(checkpoint_path.format(epoch=initial))
    
    # Entrenar el modelo
    model.fit(X_train,
              y_train,
              batch_size = 100,
              epochs = ep,
              callbacks = [cp_callback],
              validation_data = (X_test,y_test),
              verbose = 2,
              initial_epoch = initial)
        
    return model

# Programa Principal
if __name__ == "__main__":
    # Cargar y separar los datos en train y test
    X_train, X_test, y_train, y_test = loadData('X_Y_Data.pickle')

    # Crear el modelo CNN
    model = createModel(X_train.shape[1:], 66)

    # Entrenar el modelo con 1000 epocas
    model = trainModel(model, X_train, y_train, X_test, y_test, 1000)
