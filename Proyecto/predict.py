# Importar las librerias necesarias para predecir una nueva imagen
from sys import argv
from glob import glob
from scipy import misc
import numpy as np
import random
from segmentation import *
import json
from MER_NN import SymbolRecognition
from MinimumSpanningTree import MinimumSpanningTree
from os.path import isfile, join, basename
from os import listdir, getcwd, sep
import tensorflow as tf
from partition import Partition
from classifyEq import Classify

# Cragar los valores de cada simbolo requerido ("0": "dot", "1": "(", ...)
mapeo_simbolos = {}
with open('symbol_mapping.json', 'r') as opened:
	mapeo_simbolos = json.loads(opened.read())

# Definir una clase del simbolo predecido
class SymPred():
    # Definir un constructor para la clase
	def __init__(self, prediccion, x1, y1, x2, y2):
		# Definir los atributos:
        # Prediccion (latex)
        # Esquina Superior Izquierda (coordenada X)
        # Esquina Superior Izquierda (coordenada Y)
        # Esquina Inferior Derecha (coordenada X)
        # Esquina Inferior Derecha (coordenada Y)
		self.prediccion = prediccion
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2

    # Definir un metodo para obtener los atributos de la clase
	def __str__(self):
        # Retornar los atributos de la clase
		return self.prediccion + '\t' + '\t'.join([str(self.x1), str(self.y1), str(self.x2), str(self.y2)])

# Definir una clase de la imagen predecida
class ImgPred():
    # Definir un constructor para la clase
	def __init__(self, nombre_imagen, lista_simbolos, latex = 'LATEX_REPR'):
		# Definir los atributos:
        # Nombre de la imagen con extension
        # Expresion (latex)
        # Lista de simbolos predecidos
		self.nombre_imagen = nombre_imagen
		self.latex = latex
		self.lista_simbolos = lista_simbolos

    # Definir un metodo para obtener los atributos de la clase
	def __str__(self):
		res = self.nombre_imagen + '\t' + \
		    str(len(self.lista_simbolos)) + '\t' + self.latex + '\n'
		for sym_pred in self.lista_simbolos:
			res += sym_pred[0]
			res += "\t"
			res += str(sym_pred[3])
			res += "\t"
			res += str(sym_pred[1])
			res += "\t"
			res += str(sym_pred[4])
			res += "\t"
			res += str(sym_pred[2])
			res += "\n"
		return res

# Definir un metodo para predecir una nueva imagen
def predict(ruta_imagen, sess, sr):
    # Mostrar la ruta de la imagen
    print(ruta_imagen)

    # Segmentar la imagen
    seg = Segmentation(ruta_imagen)

    # Obtener las etiquetas de la iamgen
    d = seg.get_labels()

    # Mostrar las etiquetas de la imagen
    print(d)

    # Mostrar cada etiqueta y exportar imagenes segmentadas
    for label in seg.labels.keys():
        print(label)
        stroke = seg.get_stroke(label)
        scipy.misc.imsave("./tmp/"+ str(label) + ".png", stroke)

    # Aplicar el algoritmo MST
    mst = MinimumSpanningTree(d).get_mst()

    # Mostrar los pesos de la salida de MST
    print()
    print(mst)

    # Participnar las imagenes segmentadas de acuerdo al MST
    pa = Partition(mst, seg, sess, sr)

    # Obtener la lista de los simbolos
    l = pa.getList()

    # Crear un objeto para clasificar
    c = Classify()

    # Clasisifcar los simbolos
    result = c.classify(l)

    # Crear un objeto para tener los atributos de la imagen predecida
    img_prediccion = ImgPred(basename(ruta_imagen), l, result[1])

    # Retornar el objeto con los atributos asignados
    return img_prediccion

# Programa Principal
if __name__ == '__main__':
    # Obtener la ruta del modelo
    ruta_modelo = join(getcwd(), "model", "model.ckpt")

    # Definir la ruta de la imagen a predecir
    #image_folder_path = argv[1]
    image_folder_path = "./equations2"
    isWindows_flag = False
    
    # Verificar de que forma esta la ruta de las imagenes
    if len(argv) == 3:
        isWindows_flag = True
    if isWindows_flag:
        ruta_imagens = glob(image_folder_path + '\\*png')
    else:
        ruta_imagens = glob(image_folder_path + '/*png')

    # Inicializar una lista vacia de imagenes predecidas
    results = []

    # Crear sesiones de Tensor Flow para cada imagen
    with tf.Session() as sess:
        sr = SymbolRecognition(sess, ruta_modelo, trainflag = False)
        for ruta_imagen in ruta_imagens:
            # Predecir la imagen
            impred = predict(ruta_imagen, sess, sr)

            # Agregar la imagen predecida a la lista
            results.append(impred)

    # Escribir las predicciones en un archivo .txt
    with open('prediccions.txt','w') as fout:
        for res in results:
            fout.write(str(res))
