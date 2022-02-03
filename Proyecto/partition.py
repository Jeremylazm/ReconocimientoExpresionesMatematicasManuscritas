# TODO
# input mst
# ----------
# dictionary :  key = vertex, value = a list of tuples (tuple: [connected vertex, weight])
# output lst
# ----------
# lists :  list of format [symbol, y1, y2, x1, x2, list of labels]
from segmentation import Segmentation
from MinimumSpanningTree import MinimumSpanningTree
from collections import deque, defaultdict
import imghdr  # recognize img type
from os import listdir, getcwd, sep
from os.path import isfile, join
from subprocess import call
import re
import json
import tensorflow as tf
from scipy import misc
import numpy as np
from MER_NN import SymbolRecognition
from skimage.morphology import binary_dilation,dilation,disk
import scipy
from PIL import Image

symMap = {}
#abre el archivo Symbol_mapping en formato json 
with open('symbol_mapping.json', 'r') as opened:
    #lee el archivo Symbol_mapping y los guarda en symMap
    symMap = json.loads(opened.read())
print(symMap)

class Partition(object):
    def __init__(self, mst, seg, sess, sr):
        self.mst = mst
        self.seg = seg
        self.sess = sess
        self.sr = sr
        self.lst = []
        self.generateList()
        self.count = defaultdict(lambda:0)
    #retorna la lista
    def getList(self):
        return self.lst

    def calculateCount(self):
        for e in self.lst:
            self.count[e[0]]+=1
    #retorna la cantidad
    def getCount(self):
        return self.count

    def generateList(self):
        generated = []
        dots = []
        visited = set([])
        #verifica si la longitud de la lista es igual a 0
        if len(self.mst)==0:
            return
        #
        for e in self.mst:
            queue = deque([e])
            break
        while len(queue)>0:
            v = queue.popleft()
            visited.add(v)
            image = self.seg.get_combined_strokes([v])
            bb = self.seg.get_combined_bounding([v])
            image = self.input_wrapper_arr(image)
            # test = self.sr.pr(image)
            p = self.sr.p(image)
            # probability = self.sess.run(tf.nn.softmax(test)[0][0][0][p[0]])
            p = symMap[str(p[0])]
            # print probability,p
            # if probability>0. :
            self.lst.append([p,bb[0],bb[1],bb[2],bb[3],[v]])
            generated.append(v)
            if p=="-":
                # print p,bb
                pass
            #se define la multiplicacion
            elif p=="dot":
                # print "dot case"
                self.lst.pop()
                dots.append(v)
                if len(dots)>1 and len(self.lst)>0 and self.lst[-1][0]=="-":
                    l = [dots[-2],self.lst[-1][-1][0],v]
                    image = self.seg.get_combined_strokes(l)
                    bb = self.seg.get_combined_bounding(l)
                    image = self.input_wrapper_arr(image)
                    test = self.sr.pr(image)
                    p = self.sr.p(image)
                    # probability = self.sess.run(tf.nn.softmax(test)[0][0][0][p[0]])
                    p = symMap[str(p[0])]
                    # print probability,l,p
                    # if probability>0.:
                    self.lst.pop()
                    self.lst.append(["div",bb[0],bb[1],bb[2],bb[3],l])
                    dots.pop()
                    dots.pop()
                elif len(dots) == 3:
                    image = self.seg.get_combined_strokes(dots)
                    bb = self.seg.get_combined_bounding(dots)
                    image = self.input_wrapper_arr(image)
                    test = self.sr.pr(image)
                    p = self.sr.p(image)
                    # probability = self.sess.run(tf.nn.softmax(test)[0][0][0][p[0]])
                    p = symMap[str(p[0])]
                    # print probability,dots,p
                    # if probability>0.:
                    self.lst.append(["dots",bb[0],bb[1],bb[2],bb[3],dots])
                    dots = []
            #se define que es una multiplicacion
            elif p=="x" and len(self.lst)>1 and self.lst[-2][0] in ["a","b","c","d","frac"]:
                self.lst[-1][0]="mul"

            for w in self.mst[v]:
                if w[0] in visited:
                    continue
                queue.append(w[0])

        self.lst.sort(key = lambda x : x[3])
        # print self.lst
        le = len(self.lst)
        centerList = []
        minusList = []
        sList = []
        tList = []
        deleteList = []
        idx = 0
        #recorremos la lista  
        for e in self.lst:
            # guardamos en la lista centerList guardamos los datos apartir de la indice 1
            centerList.append([(e[1]+e[2])/2,(e[3]+e[4])/2])
            #si en la pocicion 0 es igual a "-" guardamos en la lista minusList
            if e[0]=="-":
                minusList.append(idx)
            #si en la pocicion 0 es igual a "s" guardamos en la lista sList
            elif e[0]=="s":
                sList.append(idx)
            #si en la pocicion 0 es igual a "t" guardamos en la lista tList
            elif e[0]=="t":
                tList.append(idx)
            idx+=1
        # recorremos en la minusList esta comineza con "-" en el cual definiremos si es una fracion  una igauldad
        for i in minusList:
            if i in deleteList:
                continue
            up = 0
            down = 0
            k=i-1
            flag = ""
            while k>=0 and centerList[k][1]<self.lst[i][4] and centerList[k][1]>self.lst[i][3]:
                if k in minusList:
                    l = self.lst[i][5]+self.lst[k][5]
                    bb = self.seg.get_combined_bounding(l)
                    deleteList.append(i)
                    deleteList.append(k)
                    self.lst.append(["=",bb[0],bb[1],bb[2],bb[3],l])
                    flag = "="
                if centerList[k][0]<centerList[i][0]:
                    up+=1
                else:
                    down+=1
                k-=1
            k=i+1
            while k<le and centerList[k][1]<self.lst[i][4] and centerList[k][1]>self.lst[i][3]:
                if k in deleteList:
                    k+=1
                    continue
                if k in minusList:
                    l = self.lst[i][5]+self.lst[k][5]
                    bb = self.seg.get_combined_bounding(l)
                    deleteList.append(k)
                    deleteList.append(i)
                    self.lst.append(["=",bb[0],bb[1],bb[2],bb[3],l])
                    flag = "="
                if centerList[k][0]<centerList[i][0]:
                    up+=1
                else:
                    down+=1
                k+=1
            if flag == "=":
                continue
            else:
                if up>0:
                    self.lst[i][0] = "frac"
                elif down>0:
                    self.lst[i][0] = "bar"
        # recorremos en la sList esta comineza con "s" en el cual definiremos si es una funcion coseno o seno
        for i in sList:
            if i in deleteList:
                continue
            k=i-1
            #la funcion coseno si k=i-1 la pocicion actual se encuentra "s", en k se encuentra "o"  y en k-1 se encuentra "c" se trata de la funcion coseno
            if k>0 and self.lst[k][4]<self.lst[k+1][3] and self.lst[k][0]=="o" and self.lst[k-1][0]=="c":
                # to the left of i
                deleteList.append(i)
                deleteList.append(i-1)
                deleteList.append(i-2)
                l = self.lst[i][5]+self.lst[i-1][5]+self.lst[i-2][5]
                bb = self.seg.get_combined_bounding(l)
                self.lst.append(["cos",bb[0],bb[1],bb[2],bb[3],l])
            #la funcion seno si en la pocion actual mas 3 encontramos a n es la funcion seno
            elif i+3<le and self.lst[i+3][0]=="n":
                # right
                deleteList.append(i)
                deleteList.append(i+1)
                deleteList.append(i+2)
                deleteList.append(i+3)
                l = self.lst[i][5]+self.lst[i+1][5]+self.lst[i+2][5]+self.lst[i+3][5]
                bb = self.seg.get_combined_bounding(l)
                self.lst.append(["sin",bb[0],bb[1],bb[2],bb[3],l])
        # recorremos en la tList esta comineza con "t" en el cual definiremos si es una funcion tagente
        for i in  tList:
            if i in deleteList:
                continue
            if i+2<le and self.lst[i+1][0]=="a" and self.lst[i+2][0]=="n":
                deleteList.append(i)
                deleteList.append(i+1)
                deleteList.append(i+2)
                l = self.lst[i][5]+self.lst[i+1][5]+self.lst[i+2][5]
                bb = self.seg.get_combined_bounding(l)
                self.lst.append(["tan",bb[0],bb[1],bb[2],bb[3],l])

        dL = sorted(deleteList,reverse=True)
        for e in dL:
            del self.lst[e]

    def input_wrapper_arr(self,image):
        #obtenemos las dimenciones de la imagen
        sx,sy = image.shape
        #obtenemos la diferencia de las dimenciones de la imagen 
        diff = np.abs(sx-sy)
        sx,sy = image.shape
        
        image = np.pad(image,((sx//8,sx//8),(sy//8,sy//8)),'constant')
        if sx > sy:
            image = np.pad(image,((0,0),(diff//2,diff//2)),'constant')
        else:
            image = np.pad(image,((diff//2,diff//2),(0,0)),'constant')
        #hace un dilatacion a la imagen
        image = dilation(image)#, footprint= disk(max(sx,sy)/32))
        #image = np.array(Image.fromarray(image).resize((32,32), Image.BILINEAR))
        #redimenciona la imagen
        image = misc.imresize(image,(32,32))
        #binariza la imagen 
        if np.max(image) > 1:
            image = image/255.
        return image


if __name__ == '__main__':
    model_path = join(getcwd(), "model", "model.ckpt")
    with tf.Session() as sess:
        sr = SymbolRecognition(sess, model_path, trainflag=False)
        imgFolderPath = getcwd() + sep + "equations"
        files = [f for f in listdir(imgFolderPath) if isfile(join(imgFolderPath, f)) and imghdr.what(join(imgFolderPath, f))=='png']
        for fname in files:
            # fname='./equations/SKMBT_36317040717260_eq16.png'
            print(fname)
            seg = Segmentation(join(imgFolderPath, fname))
            d = seg.get_labels()
            mst = MinimumSpanningTree(d).get_mst()
            pa = Partition(mst,seg,sess,sr)
            print(pa.getList())
            pa.calculateCount()
            print(pa.getCount())
            # for label in seg.labels.keys():
            #     # print label
            #     stroke = seg.get_stroke(label)
            #     scipy.misc.imsave('./tmp/'+ str(label)+'.png', stroke)
