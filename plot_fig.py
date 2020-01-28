from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from random import randint
import matplotlib.pyplot as plt

import scikitplot as skplt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.spatial import distance

face2_igual = np.load('arrays_saida/data_face2_igual_270120201652.npy') # load
face2_outro = np.load('arrays_saida/data_face2_outro_270120201652.npy') # load
face_igual = np.load('arrays_saida/data_face_igual_2016.npy') # load
face_outro = np.load('arrays_saida/data_face_outro_2016.npy') # load

def plot_roc_curve(igual, outro, cor):
    
    #preparando os dados
    array = []

    for i in range(len(igual)):
        array.append(igual[i])

    for i in range(len(outro)):
        array.append(outro[i])
  
    # Plotando curva ROC

    # criando vetor de zeros
    array_f = np.zeros(len(array))

    # preenchendo o vetor criado com 1's até o tamanho do vetor com as predições verdadeiras
    for i in range(len(igual)):
        array_f[i] = 1

    a, b, c = metrics.roc_curve(array_f, array)
    faceAuc = auc(b, a)
    
    plt.plot(b, a, linestyle='--', color=cor, label='\nauc: {}' .format(faceAuc)) #plota as rocs com as aucs

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # show the legend
    plt.legend()


def plot_histogram(face_igual, face_outro, vgg_igual, vgg_outro, cor1, cor2):
    plt.subplot(1, 2, 1)
    plt.hist(face_igual, 30, alpha = 0.8, facecolor=cor1)
    plt.hist(face_outro, 30, alpha = 0.8, facecolor=cor2)
    plt.title('VGGFACE - euclidean')
    plt.xlabel('distancia euclidiana')
    plt.ylabel('QTD ocorrencias')

    plt.subplot(1, 2, 2)
    plt.hist(vgg_igual, 30, alpha = 0.8, facecolor=cor1)
    plt.hist(vgg_outro, 30, alpha = 0.8, facecolor=cor2)
    plt.title('VGGFACE2 - euclidean')
    plt.xlabel('distancia euclidiana')
    plt.ylabel('QTD ocorrencias')
    plt.show()


vermelha = 'red'
verde = 'green'
azul = 'blue'
amarelo = 'yellow'
laranja = 'orange'

plot_histogram(face_igual, face_outro, face2_igual, face2_outro, vermelha, azul)
plot_roc_curve(face_igual, face_outro, verde)
plot_roc_curve(face2_igual, face2_outro, azul)
plt.show()