# Las siguientes funciones recorren 80000 imágenes de cada caracter, y guardan la data en una matriz de 3D de 80,000 x 45 x 45. Luego guarda la data en un archivo pickle.

# Librerías
import os
import numpy as np
import imageio
import csv
import sys
from sklearn.model_selection import train_test_split
import cv2
import pickle

def crearDict(ruta_imagenes) -> None:

    # Recorrer las carpetas del directorio
	listadir = os.listdir(ruta_imagenes)

    # único -> nombres de las carpetas de un solo caracter
    # multiple -> nombres de las carpetas de varios caracteres
	unico = []
	multiple = []

    # Agregar a único o multiple según corresponda
	for dir in listadir:
		dir = dir.lower() # Convertir a minúsculas
		if len(dir) == 1:
			unico.append(dir)
		else:
			multiple.append(dir)

	multiple.sort() # Ordenar alfabéticamente

	unico.sort() # Ordenar con el orden ascii

    # Diccionarios de caracteres
	dict = {}
	contador = 0

    # Crear diccionario
	for dir in multiple:
		dict[dir] = contador
		contador += 1

	for dir in unico:
		dict[dir] = contador
		contador += 1

	# Guardarlo en un archivo csv
	archivo = open("LabelDict.csv", "w")
	w = csv.writer(archivo)

	for clave, valor in dict.items():
		w.writerow([clave,valor])

    # Cerrar archivo
	archivo.close()

# Cargar diccionario de caracteres (clave, valor)
def cargarDict_AB(nombre_archivo) -> dict:
	dict = {}
	with open(nombre_archivo) as f:
		leerCSV = csv.reader(f)
		for fila in leerCSV:
			if len(fila) > 0:
				dict[fila[0]] = int(fila[1])
	return dict

# Cargar diccionario de caracteres (valor, clave)
def cargarDict_BA(nombre_archivo) -> dict:
	dict = {}
	with open(nombre_archivo) as f:
		leerCSV = csv.reader(f)
		for fila in leerCSV:
			if len(fila) > 0:
				dict[int(fila[1])] = fila[0]
	return dict

# Cargar dataset de imágenes
# nombre_archivo1 -> ubicación de todas las imágenes
# nombre_archivo2 -> ubicación del diccionario
def cargarDataset(nombre_archivo1, nombre_archivo2, rate = 0.2):
	dict = cargarDict_AB(nombre_archivo2)
	ds1 = os.listdir(nombre_archivo1)
	contador_archivo = sum([len(files) for r, d, files in os.walk(nombre_archivo1)])
	contador = 0

	X = np.empty((0, 45, 45), dtype = np.uint8)
	Y = np.empty((0, 1), dtype = np.uint8) 

	for d in ds1:
		folder = os.path.join(nombre_archivo1, d)
		ds2 = os.listdir(folder)
		d = d.lower()
		for d2 in ds2:
			filei = os.path.join(folder,d2)
			imagen = cv2.imread(filei)

            # Convertir a escala de grises
			image = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) 
            
            # Redimensionar a 45x45
			npi = np.asarray(image).reshape(45,45)

			X = np.append(X, [npi], axis = 0)
			Y = np.append(Y, dict[d])

			contador += 1
			output_string = f"Archivo de imagen {contador} de {contador_archivo}\n"

			sys.stdout.write(output_string)
			sys.stdout.flush()

	return X, Y

if __name__ == '__main__':
    # Ruta de las imágenes
	ruta = './extracted_images/'
	crearDict(ruta)
	
    # Nombre del diccionario de caracteres
	nombre_diccionario = 'LabelDict.csv'
	dict = cargarDict_BA(nombre_diccionario)

	X, Y = cargarDataset(ruta, nombre_diccionario, rate = 0.2)

    # Serializar X y Y en un archivo pickle
	with open('X_Y_Data.pickle', 'wb') as f:
		pickle.dump([X, Y], f)