import scipy
from scipy import misc
from scipy import ndimage
import numpy as np
import sys
import datetime
from skimage import color

class Segmentation(object):

    def __init__(self, path):
        img = misc.imread(path)
        self.origin = img
        # Aplicar filtro gaussiano para eliminar ruido y suavizar la imagen
        blur_radius = 1.0
        imgf = ndimage.gaussian_filter(img, blur_radius)
        # Encontrar componentes (número de caracteristicas)
        # labeled_img es una imagen etiquetada de acuerdo a num_features
        threshold = 50
        self.labeled_img, self.num_features = ndimage.label(imgf > threshold)
        # Obtener las coordenadas del cuadro delimitador para cada componente etiquetado
        self.labels = self.calculate_labels()

    def get_labels(self):
        return self.labels
    
    def calculate_labels(self):
        '''
        Devuelve las coordenadas del cuadro delimitador para cada componente etiquetado de la imagen.    
        :return: Diccionario de etiquetas para cada componente y las coordenadas de su cuadro delimitador.
                 Coordenadas: [y_min, y_max, x_min, x_max]
        '''
        labels = {}
        for label in range(1, self.num_features + 1):
            labels[label] = [sys.maxsize, -1, sys.maxsize, -1] # [y_min, y_max, x_min, x_max]

        # Calcular las coordenadas del cuadro delimitador para cada componente
        img = self.labeled_img
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                value = img[row][col]
                if value != 0:
                    bounding = labels[value]
                    if bounding[0] > row:
                        bounding[0] = row
                    if bounding[2] > col:
                        bounding[2] = col
                    if bounding[1] < row:
                        bounding[1] = row
                    if bounding[3] < col:
                        bounding[3] = col
                        
        # Eliminar los componentes donde el área del cuadro delimitador es menor a 50
        to_del = []
        for label in labels:
            bounding = labels[label]
            if (bounding[1] - bounding[0]) * (bounding[3] - bounding[2]) < 50:
                to_del.append(label)
        for key in to_del:
            del labels[key]
            
        return labels

    def get_stroke(self,label):
        '''
        Devuelve el trazo de un componente etiquetado     
        :param label: Diccionario. key = etiqueta del componente, values: lista de coordenadas del cuadro delimitador
        :return: Trazo del componente.
        '''
        # Obtener las coordenadas del cuadro delimitador
        l = self.labels[label] # [y_min, y_max, x_min, x_max]
        # Obtener el trazo a partir de las coordenadas del cuadro delimitador en la imagen etiquetada
        y_min = l[0]
        y_max = l[1]
        x_min = l[2]
        x_max = l[3]
        stroke = np.copy(self.labeled_img[y_min:y_max + 1, x_min:x_max + 1])
        shape = stroke.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if stroke[i][j] != label:
                    stroke[i][j] = 0 # Eliminar componentes que se traslapan dentro del cuadro delimitador
                else:
                    stroke[i][j] = self.origin[i+y_min][j+x_min] # Obtener pixeles originales
        
        return stroke

    def get_combined_strokes(self,l_labels):
        '''
        Devuelve el conjunto de componentes etiquetados en un único trazo
        :param l_labels: Diccionario del conjunto de componentes etiquetados
        :return: Trazo único del conunto de componentes etiquetados
        '''
        # Obtener las coordenadas del cuadro delimitador del conjunto total de componentes
        bounding = self.get_combined_bounding(l_labels) # [y_min, y_max, x_min, x_max]
        # Obtener el trazo a partir de las coordenadas del cuadro delimitador en la imagen etiquetada
        y_min = bounding[0]
        y_max = bounding[1]
        x_min = bounding[2]
        x_max = bounding[3]
        stroke = np.copy(self.labeled_img[y_min:y_max + 1, x_min:x_max + 1])
        shape = stroke.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if stroke[i][j] in l_labels:
                    stroke[i][j] = self.origin[i+y_min][j+x_min] # Obtener pixeles originales
                else:
                    stroke[i][j] = 0

        return stroke

    def get_combined_bounding(self,l_labels):
        '''
        Devuelve las coordenadas del cuadro delimitador de un conjunto de componentes etiquetados
        :param l_labels: Diccionario del conjunto de componentes etiquetados
        :return: Cuadro delimitador del conjunto de componentes etiquetados
        '''
        l = [sys.maxsize,-1,sys.maxsize,-1] # Coordenadas del cuadro delimitador 
        for label in l_labels:
            bounding = self.labels[label]
            if bounding[0] < l[0]:
                l[0] = bounding[0]
            if bounding[1] > l[1]:
                l[1] = bounding[1]
            if bounding[2] < l[2]:
                l[2] = bounding[2]
            if bounding[3] > l[3]:
                l[3] = bounding[3]

        return l

    def draw_boundings(self, width):
        '''
        Devuelve el cuadro delimitador de todos los componentes en la imagen original
        :param width: Ancho de la linea del cuadro delimitador
        :return: Imagen original con los cuadros delimitadores de todas los componentes etiquetados
        '''
        comp_boxes = color.gray2rgb(self.origin)
        green = np.array([0, 255, 0], dtype=np.uint8)
        w = width # ancho de la linea
        for label in self.labels.keys(): 
            coord = self.labels[label] # Coordenadas del cuadro delimitador de cada componente
            comp_boxes[coord[0] - w:coord[0], coord[2] - w:coord[3] + w] = green
            comp_boxes[coord[1]:coord[1] + w, coord[2] - w:coord[3] + w] = green
            comp_boxes[coord[0]:coord[1], coord[2] - w:coord[2]] = green
            comp_boxes[coord[0]:coord[1], coord[3]:coord[3] + w] = green
        
        return comp_boxes

if __name__ == '__main__':
    fname = 'annotated/SKMBT_36317040717280_eq3.png'
    seg = Segmentation(fname)

    print(seg.labels)

    for label in seg.labels.keys():
        print(label)
        stroke = seg.get_stroke(label)
        scipy.misc.imsave("./tmp/"+ str(label) + ".png", stroke)

    origin = seg.origin
    scipy.misc.imsave('./tmp/origin.png', origin)

