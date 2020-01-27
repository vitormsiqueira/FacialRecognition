#Obs: Boa curva de comparaçãp: Curva Roc (pesquisar) 

#Tarefa pra casa:
    #Métodos de Score

# face verification with the VGGFace2 model
from PIL import Image
from glob import glob
from random import randint
import os
from itertools import combinations
from random import randint
from numpy import asarray
from scipy.spatial import distance
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from keras.models import Model, Sequential
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras.applications.vgg16 import VGG16
from keras_vggface.utils import preprocess_input
from keras_vggface import utils
import numpy as np
from keras.preprocessing import image
from sklearn.preprocessing import normalize
import random

direc_dataset = "/media/vitor/SHARE/DEV/Visão Computacional/dataset/lfw_eleicao_test1" #pasta contendo a pasta de cada pessoa
date_save = "240120201520" # adicionar aqui a data e hora de realização do teste para salvar os dados correspondentes

def euclideanDistance(base, teste):
    var = base - teste
    var = np.sum(np.multiply(var, var))
    var = np.sqrt(var)
    return var

def cosineDistance(base, teste):
    var = distance.cosine(base, teste)
    return var

# create a vggface2 model
vggface2_model = VGGFace(model='resnet50', include_top=True)
vggface2_model.summary()
vggface2_model = Model(inputs=vggface2_model.layers[0].input, outputs=vggface2_model.layers[-3].output)

#create a vggface model

# vggface_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
# # vggface_model.summary()
# vggface_model = Model(inputs=vggface_model.layers[0].input, outputs=vggface_model.layers[-2].output)

#create a vgg16 model
# vgg16_model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# vgg16_model.summary()

face2_igual = [] #distancia para imagens da mesma pessoa em vggFace2
# face_igual = [] #distancia para imagens da mesma pessoa em vggFace

face2_outro = [] #distancia entre pessoa e as outras em vggFace2
# face_outro = [] #distancia entre pessoa e as outras em vggFace


def read_image(nome, direct_by_person):

    path_photo = direct_by_person+"/"+nome

    img = image.load_img(path_photo, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)
    return x


def veri_face(photo1, photo2, direct_by_person, direct_by_person2):

    # preds1 = vggface_model.predict(read_image(photo1, direct_by_person))
    # preds2 = vggface_model.predict(read_image(photo2, direct_by_person2))

    preds11 = vggface2_model.predict(read_image(photo1, direct_by_person))
    preds22 = vggface2_model.predict(read_image(photo2, direct_by_person2))

    # distancia_vgg = euclideanDistance(preds1, preds2)
    distancia_vgg2 = euclideanDistance(preds11, preds22)

    return distancia_vgg2


def dif_person(photo_1, photo_2, name_photo_1, name_photo_2, direc_dataset):
    
    direct_by_person_1 = direc_dataset+"/"+name_photo_1
    direct_by_person_2 = direc_dataset+"/"+name_photo_2

    dist_vgg2 = veri_face(photo_1, photo_2, direct_by_person_1, direct_by_person_2)

    face2_outro.append(dist_vgg2)

    print("dist vgg2 ->", dist_vgg2)

            
def same_person(path_person_list, name_person, direc_dataset):

    print("\nSame Person = "+name_person)
    
    list_comb = list(combinations(path_person_list, 2)) #gera lista de combinações entre duas fotos da mesma pessoa
    direct_by_person = direc_dataset+"/"+name_person
    len_list_comb = len(list_comb)
    print(len_list_comb)

    print("VggFace\n")

    cont = 0 # contador de quantas combinações entre duas imagens iguais foram realizadas

    for i in range(len_list_comb):

        cont = i

        photo1, photo2 = list_comb[i]

        dist_vgg2 = veri_face(photo1, photo2, direct_by_person, direct_by_person) # retorna a distancia euclidiana entre fotos da mesma pessoa
        # face_igual.append(dist_vgg)
        face2_igual.append(dist_vgg2)

        print("dist vgg2", dist_vgg2)

    return cont


def ger_pessoas(direc_dataset):
    
    all_persons = os.listdir(direc_dataset) # recebe o nome de cada pasta, onde se encontra cada pessoa
    name_person = [] #cria uma lista onde iremos armazenar o nome de cada pessoa (nome da pasta)
    print(all_persons)
    
    # print("\n=============== *Same peoples* =================")

    # for i in range(len(all_persons)): # define quantas pastas serão percorridas
    #     name_person = all_persons[i] # a lista name_person recebe como parametro o nome correspondente na lista all_persons
        
    #     path_person_list = os.listdir(direc_dataset+"/"+name_person) # lista o diretorio da pessoa
    #     len_by_person = len(path_person_list) # guarda a quantidade de fotos contida em sua respectiva pasta

    #     if len_by_person > 1:
    #         cont = same_person(path_person_list, name_person, direc_dataset)

    # # Save the data as array 
    # data_face2_igual = np.asarray(face2_igual)
    # np.save('/media/vitor/SHARE/DEV/Visão Computacional/siamese_net/data_face2_igual_'+date_save, data_face2_igual) # salva os arrays de distancia

    print("\n=============== *Different peoples* =================")

    array_all_pic = [] # array contendo todas as fotos    

    # aqui junto todas as fotos do dataset em um único vetor
    for i in range(len(all_persons)): # pode demorar muito dependendo da quantidade de combinações
        name_person = all_persons[i]
        path_person_list = os.listdir(direc_dataset+"/"+name_person) # lista o diretorio da pessoa

        array_all_pic.append(path_person_list)

    print(array_all_pic)
    print(array_all_pic[0])
    print(array_all_pic[3])

    # name_photo_1 = photo_1[:-9] # fatia a string eliminando o últimos 9 caracteres, que contem o nome da pasta de cada pessoa, restando apenas o nome
    # number_photo_1 = photo_1[-8:-4] # fatia a string, gravando apenas o número da foto

    # print(name_photo_1)
    # print(number_photo_1)

    # Dict[name_photo_1] = number_photo_1

    # list_all_compara = []

    # for i in range(20):
        
    #     photo_1_pos = randint(0, len(array_all_pic)-1) # recebe um numero aleatorio que indicará qual imagem (na lista de todas as imagens) iremos usar para a comparação
    #     photo_1 = array_all_pic[photo_1_pos] # a photo_1 vai ser a photo que se encontra na posição do numero aleatorio no array que contem todas as fotos

    #     photo_2_pos = randint(0, len(array_all_pic)-1) # recebe um numero aleatorio que indicará qual imagem (na lista de todas as imagens) iremos usar para a comparação
    #     photo_2 = array_all_pic[photo_2_pos] # a photo_1 vai ser a photo que se encontra na posição do numero aleatorio no array que contem todas as fotos

    #     list_all_compara.append()
        

    #     print(list_all_compara)

        # photo_2_pos = randint(0, len(array_all_pic)-1)

        # print(photo_1_pos)
        # print(photo_2_pos)

        # photo_1 = array_all_pic[photo_1_pos] # a photo_1 vai ser a photo que se encontra na posição do numero aleatorio no array que contem todas as fotos
        # photo_2 = array_all_pic[photo_2_pos]

        # name_photo_1 = photo_1[:-9] # fatia a string eliminando o últimos 9 caracteres, que contem o nome da pasta de cada pessoa, restando apenas o nome
        # name_photo_2 = photo_2[:-9]

        # print("Pessoa 1: "+name_photo_1)
        # print("Pessoa 2: "+name_photo_2)


    #     dif_person(photo_1, photo_2, name_photo_1, name_photo_2, direc_dataset)

    # print("\nLista VGGFace2 de faces iguais\n", face2_igual)
    # # print("Lista VGGFace de faces iguais\n", face_igual)

    # print("\nLista VGGFace2 de faces diferentes\n", face2_outro)
    # # print("Lista VGGFace2 de faces diferentes\n", face_outro)

    # # data_face_igual = np.asarray(face_igual)
    # # data_face_outro = np.asarray(face_outro)

    # # Save the data as array 
    # data_face2_outro = np.asarray(face2_outro)

    # print(cont)
    
    # np.save('/media/vitor/SHARE/DEV/Visão Computacional/siamese_net/data_face_igual_01121625', data_face_igual)
    # np.save('/media/vitor/SHARE/DEV/Visão Computacional/siamese_net/data_face_outro_01121625', data_face_outro)
    # np.save('/media/vitor/SHARE/DEV/Visão Computacional/siamese_net/data_face2_outro_'+date_save, data_face2_outro)


ger_pessoas(direc_dataset)

