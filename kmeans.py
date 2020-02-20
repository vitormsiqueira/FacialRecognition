import numpy as np
import os
from keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from keras.models import Model, Sequential
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
from pandas import DataFrame
import pandas
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


direc_dataset = "/media/vitor/SHARE/DEV/Visão Computacional/dataset/lfw_eleicao_test1/" #pasta contendo a pasta de cada pessoa
qtd_total_photos = 495
qtd_pessoas = 21

def create_model():
    vggface2_model = VGGFace(model='resnet50', include_top=True)
    # vggface2_model.summary()
    vggface2_model = Model(inputs=vggface2_model.layers[0].input, outputs=vggface2_model.layers[-3].output)
    
    return vggface2_model

def load_data():
    features_face2 = np.load('/media/vitor/SHARE/DEV/Visão Computacional/siamese_net/kmeans/006data_face2_features.npy') # load
    array_kmeans = np.load('/media/vitor/SHARE/DEV/Visão Computacional/siamese_net/kmeans/006data_kmeans.npy') # load

    # print(features_face2.shape)

    return features_face2, array_kmeans

def prepare_img(posicao_na_pasta, name):
    
    # Agora vamos acessar a imagem

    pessoa = direc_dataset+'/'+name

    # Precisamos saber qual foto representa o number_name
    own_photos = os.listdir(pessoa)
    own_photos.sort()
    # print('own photos', own_photos)
    # print('len own_photos', len(own_photos))

    print(posicao_na_pasta)
    pessoa_photo = pessoa+'/'+own_photos[posicao_na_pasta]

    return pessoa_photo
    
def kmeans(features_face2, number_clusters):
    kmean = KMeans(n_clusters=number_clusters)
    kmean.fit(features_face2)

    final_clustering = kmean.predict(features_face2)

    # print(final_clustering)
    # print(len(final_clustering))
    # print(array_kmeans)

    return final_clustering

def size_by_name():

    list_label = os.listdir(direc_dataset)
    size_by_name = []
    # print(list_label)

    name = []
    # Esse laço retorna um vetor onde cada posição tem o n de fotos da pasta atual + o n de fotos da pasta anterior 
    for i in range(len(list_label)):
        name = list_label[i]
        # print(name)
        direct_name = os.listdir(direc_dataset+'/'+name)
        size_by_name.append(len(direct_name))

    # print(size_by_name)

    return size_by_name, list_label


def cluster_analysis():
    
    number = 0
    array_all_class = []
    
    # Esse array vai salvar todas as classes que serão definidas por pessoa 
    current_class = []

    # Esse laço separa a análise por pessoa (i)
    for i in range(len(size_by_name)):
        for f in range(size_by_name[i]):
            # print(f)
            print(f+1, '--', final_clustering[f+number], '--', f+number)

            array_all_class.append(final_clustering[f+number])
        
        # Aqui descobrimos qual classe representa a pessoa (analisando a classe que mais de repete) e adicionamos em uma lista
        counts = np.bincount(array_all_class)
        current_class.append(np.argmax(counts))
        
        print('Classe da pessoa', current_class[i])
        print('print array_all_class', array_all_class)
        
        # Agora verificamos o array da pessoa em busca uma classe que não a descreve e pegar a posição dessa foto "ruim"
        for posicao in range(len(array_all_class)):
            
            # Verificamos se a classe atual da pessoa é diferente da classe geral que foi definida
            if (array_all_class[posicao] != current_class[i]):
                
                # print('array_all_class[posicao]',array_all_class[posicao])
                # print('current_class[posicao]', current_class[i])
                
                # Aqui vamos acessar a pasta da pessoa e mostrar a foto que não a representa
                name = list_label[i]

                # Aqui usamos essa função para mostrar as imagens "ruins"
                pessoa_photo = prepare_img(posicao, name)

                img=mpimg.imread(pessoa_photo)
                imgplot = plt.imshow(img)
            plt.show()


        # Deletamos o array_all_class
        del(array_all_class[:])
        
        print("------------", f+1)
        number += (f+1) #pula pra próxima posição


################## MAIN ######################

# 1º criar o modelo
vggface2_model = create_model()

# 2º carregar os dados
features_face2, array_kmeans = load_data()

# 3º Aplicar Reshape no vetor de caracteristicas 
features_face2 = np.reshape(features_face2,(qtd_total_photos, 2048))

# 4º Aplicar a Clusterização
final_clustering = kmeans(features_face2, qtd_pessoas) 

# 5º Listar todas as pessoas do dataset
all_people = os.listdir(direc_dataset)

# 6º Saber a qtdd de fotos por pessoa
size_by_name, list_label = size_by_name()

# 7º Analisar pessoa por pessoa para encontrar as fotos que não a representam
cluster_analysis()

##############################################
