from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from random import randint
import matplotlib.pyplot as plt

import scikitplot as skplt

from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.spatial import distance

face2_igual = np.load('arrays_saida/data_face2_igual_2016.npy') # load
face2_outro = np.load('arrays_saida/data_face2_outro_2016.npy') # load
face_igual = np.load('arrays_saida/data_face_igual_2016.npy') # load
face_outro = np.load('arrays_saida/data_face_outro_2016.npy') # load

# print(len(face2_igual))
# print(len(face2_outro))
# print(len(face_igual))
# print(len(face_outro))

# Plotando os histogramas

# plt.subplot(1,2,1)

# plt.xlabel("Distância Euclidiana", fontsize=15)
# plt.ylabel("Frequencia", fontsize=15)
# plt.title("VGG FACE")

# plt.hist(face_igual, facecolor='blue')
# plt.hist(face_outro, facecolor='red')

# plt.subplot(1,2,2)

# plt.xlabel("Distância Euclidiana", fontsize=15)
# plt.ylabel("Frequencia", fontsize=15)
# plt.title("VGG FACE 2")

# plt.hist(face2_igual, facecolor='blue')
# plt.hist(face2_outro, facecolor='red')

# plt.show()

# import scikitplot as skplt
# from sklearn import metrics

# Plotando curva ROC

# criando um vetor
# array_f_igual = []
# array_f2_outro = []

# concatenando os dois arrays em um unico vetor
# array_f_igual = np.concatenate((face_igual, face_outro)) 
# array_f2_igual = np.concatenate((face2_igual, face2_outro))

# print(len(face_igual), len(face_outro))
# print(len(array_f_igual))
# print(type(array_f_igual))

# criando vetor de zeros
# array_f = np.zeros(len(face_igual)+len(face_outro))
# print(len(array_f))
# array_f2 = np.zeros(len(face2_igual)+len(face2_outro))

#preenchendo o vetor criado com 1's até o tamanho do vetor com as predições verdadeiras
# for i in range(len(face2_igual)):
    # array_f[i] = 1
    # array_f2[i] = 1
    
# print(array_f)

# a, b, c = metrics.roc_curve(array_f, array_f_igual)
# d, e, f = metrics.roc_curve(array_f2, array_f2_igual)

# plt.plot(b, a)
# plt.plot(e, d)
# plt.show()

# len_face2_igual = len(face2_igual)
# len_face2_outro = len(face2_outro)

# print(array_f2)

def plot_histogram(face_igual, face_outro, vgg_igual, vgg_outro):
    plt.subplot(1, 2, 1)
    plt.hist(face_igual, 30, alpha = 0.8, facecolor='blue')
    plt.hist(face_outro, 30, alpha = 0.8, facecolor='green')
    plt.title('VGGFACE - euclidean')
    plt.xlabel('distancia euclidiana')
    plt.ylabel('QTD ocorrencias')

    plt.subplot(1, 2, 2)
    plt.hist(vgg_igual, 30, alpha = 0.8, facecolor='blue')
    plt.hist(vgg_outro, 30, alpha = 0.8, facecolor='green')
    plt.title('VGGFACE2 - euclidean')
    plt.xlabel('distancia euclidiana')
    plt.ylabel('QTD ocorrencias')
    plt.show()

#print(face_igual)

def plot_curve_roc(face_igual, face_outro, color):

    y = np.zeros(face_igual.shape[0] + face_outro.shape[0])
    scores = np.zeros(face_igual.shape[0] + face_outro.shape[0])

    # gera um vetor com as classes 1 para pessoas iguais, 0 para diferentes
    for i in range(face_igual.shape[0]):
        y[i] = 1
        scores[i]=face_igual[i]

    # coloca os dois vetores (pessoas iguais e diferentes), num mesmo vetor, em sequencia
    for i in range(face_outro.shape[0]):
        scores[face_igual.shape[0]+i] = face_outro[i]


    A,B,C = roc_curve(y,scores) #calcula os parametros para a roc usando os vetores gerados acima
    faceAuc = auc(B, A) #calcula a area auc usando  saida do roc acima

    #plt.subplot(1, 3, 3)
    plt.plot(B, A, linestyle='--', color=color, label='\nauc: {}' .format(faceAuc)) #plota as rocs com as aucs

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # show the legend
    plt.legend()
    # show the plot
    #plt.show()

vermelha = 'red'
verde = 'green'
azul = 'blue'
#plota o histograma

plot_histogram(face_igual, face_outro, face2_igual, face2_outro)
plot_curve_roc(face_igual, face_outro, verde)
plot_curve_roc(face2_igual, face2_outro, azul)
plt.show()