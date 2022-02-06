import math
from queue import Queue
from segmentation import *

class MinimumSpanningTree(object):

    def __init__(self, labels):
        self.d_mst = {}
        l_weight = []
        # Calcular los pesos entre todos los nodos (componentes etiquetados)
        for i in labels.keys():
            for j in labels.keys():
                if j > i:
                    weight = self.calculate_weight(labels[i], labels[j])
                    t = [[i, j], weight] # Peso entre dos nodos
                    l_weight.append(t)
        self.l_weight = sorted(l_weight, key=lambda x: x[1]) # Ordena los pesos ascendentemente
        #print(self.l_weight)
        self.build_tree()

    def get_mst(self):
        return self.d_mst

    def calculate_weight(self, l_bounding_1, l_bounding_2):
        '''
        Calcula la distancia (pesos) entre los cuadros delimitadores de dos componentes etiquetados
        :params l_bounding_1, l_bounding_2: Diccionario. key = etiqueta del componente, values: lista de coordenadas del cuadro delimitador
        :return: Distancia entre los dos nodos (componentes)
        '''
        y_mean_1 = (l_bounding_1[0] + l_bounding_1[1]) / 2.0
        y_mean_2 = (l_bounding_2[0] + l_bounding_2[1]) / 2.0
        x_mean_1 = (l_bounding_1[2] + l_bounding_1[3]) / 2.0
        x_mean_2 = (l_bounding_2[2] + l_bounding_2[3]) / 2.0
        # Calcular distancia euclidinana
        euclidean_distance = math.sqrt(math.pow(x_mean_1 - x_mean_2, 2) + math.pow(y_mean_1 - y_mean_2, 2))
        return euclidean_distance

    def build_tree(self):
        '''
        Devuelve la estructura del árbol de expansión mínimo (Minimum Spanning Tree)
        :return: Estructura de árbol
        '''
        d_mst = self.d_mst
        while self.l_weight:
            edge = self.l_weight.pop(0)
            v1 = edge[0][0] # Primer nodo
            v2 = edge[0][1] # Segundo nodo
            #print(d_mst)
            if self.is_disconnected(v1, v2): # Los nodos no están conectados
                # Vincular al primer nodo el nodo no conectado y su peso
                self.dictionary_add(d_mst, v1, [v2, edge[1]]) 
                self.dictionary_add(d_mst, v2, [v1, edge[1]])
        return d_mst

    def is_disconnected(self, v1, v2):
        '''
        Verifica si dos nodos no están conectados (no son conexos)
        :param v1, v2: nodos
        :return: True si los dos nodos no están conectados y False caso contrario
        '''
        d_mst = self.d_mst
        if (v1 not in d_mst.keys()) or (v2 not in d_mst.keys()):
            return True
        if v1 in self.d_mst.keys():
            return self.search(v1, v2)
        else:
            return self.search(v2, v1)

    def search(self, v1, v2):
        '''
        Verifica si el nodo v2 se encuentra conectado con uno de los nodos que están conectados al nodo v1
        :param v1, v2: nodos
        '''
        l_visited = [] # Nodos visitados
        q = Queue()
        q.put(v1)
        while q.qsize() > 0:
            cur_v = q.get()
            l_visited.append(cur_v)
            for t in self.d_mst[cur_v]:
                neighbour = t[0]
                if v2 == neighbour:
                    return False
                if neighbour not in l_visited:
                    q.put(neighbour)
        return True

    def dictionary_add(self, d, key, t):
        '''
        # Vincula un nodo del árbol de expansión mínimo con otro nodo no conectado
        :param Key: primer nodo
        :param t: Array. Segundo nodo vinculado y la distancia o peso al primer nodo
        '''
        if key not in d.keys():
            d[key] = []
        if t not in d[key]:
            d[key].append(t)
    
    def center_of_mass(self, img, coords):
        '''
        Dada una imagen, retorna el centro de masa de la imagen       
        :param img: La imagen que será procesada
        :param coords: [y_min, y_max, x_min, x_max]
        :return: El centro de masa el objeto en la imagen
        '''
        sum_x = coords[2]
        sum_y = coords[0]
        k = 0 # [y_min, y_max, x_min, x_max]
        for i in range(coords[2], coords[3]):
            for j in range(coords[0], coords[1]):
                if img[j][i] > 0:
                    sum_x += i*img[j][i]
                    sum_y += j*img[j][i]
                    k += img[j][i]

        sum_x = sum_x // k
        sum_y = sum_y // k
        return sum_x, sum_y
    
    def draw_MST(self, img, labels):
        '''
        Devuelve la estructura del MST en la imagen original
        :param: img: Imagen original en escala de grises 
        :param: labels: Dict. Componentes etiquetados
        :param: width: Ancho de la linea del árbol
        :return: Imagen original con los cuadros delimitadores de todas los componentes etiquetados
        '''
        origin = color.gray2rgb(img)
        for v in self.d_mst.keys():
            coords_v = labels[v]
            center_v = self.center_of_mass(img, coords_v)
            for v_con in self.d_mst[v]:
                coords_v_con = labels[v_con[0]] 
                center_v_con = self.center_of_mass(img, coords_v_con)
                cv2.line(origin, (center_v[0], center_v[1]), (center_v_con[0], center_v_con[1]), (0, 255, 0))
                #self.draw_line(origin, red, center_v[0], center_v[1], center_v_con[0], center_v_con[1])

        return origin

if __name__ == '__main__':
    # d = {1: [21, 150, 434, 533], 2: [26, 79, 683, 775], 3: [43, 66, 489, 523], 4: [47, 74, 735, 779], 5: [76, 86, 479, 535],
    #      6: [84, 91, 600, 625], 7: [88, 99, 678, 788], 8: [96, 104, 596, 618], 9: [98, 136, 495, 523],
    #      10: [117, 166, 683, 778], 11: [135, 171, 739, 769]}

    import imageio
    import cv2
    # Leer la imagen
    img = cv2.imread('SKMBT_36317040717260_eq16.png', cv2.IMREAD_GRAYSCALE)
    # Binarizar la imagen
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Segmentar la imagen
    seg = Segmentation(th2)
    d = seg.get_labels()
    mst = MinimumSpanningTree(d)
    img_mst = mst.draw_MST(th2, d)
    imageio.imsave('img_mst.png', img_mst.astype(np.uint8))

