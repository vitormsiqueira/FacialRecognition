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


def mostrar_img(posicao_na_pasta, name):
    
    # Agora vamos acessar a imagem

    pessoa = direc_dataset+'/'+name

    # Precisamos saber qual foto representa o number_name
    own_photos = os.listdir(pessoa)
    own_photos.sort()
    # print('own photos', own_photos)
    # print('len own_photos', len(own_photos))

    print(posicao)
    pessoa_photo = pessoa+'/'+own_photos[posicao_na_pasta]

    print(pessoa_photo)
    
    # Listamos todas as fotos e armazenamos em um array
    
    # path_person_list = os.listdir(pessoa)
    # array_all_photos.append(path_person_list)


vggface2_model = VGGFace(model='resnet50', include_top=True)
# vggface2_model.summary()
vggface2_model = Model(inputs=vggface2_model.layers[0].input, outputs=vggface2_model.layers[-3].output)

features_face2 = np.load('/media/vitor/SHARE/DEV/Visão Computacional/siamese_net/kmeans/006data_face2_features.npy') # load
array_kmeans = np.load('/media/vitor/SHARE/DEV/Visão Computacional/siamese_net/kmeans/006data_kmeans.npy') # load

print(features_face2.shape)

direc_dataset = "/media/vitor/SHARE/DEV/Visão Computacional/dataset/lfw_eleicao_test1/" #pasta contendo a pasta de cada pessoa

array_label = []

# for i in range(501):
#     array_final.append(features_face2[i])

# print(array_final)
# print(len(array_final))
# print(type(features_face2))

direc_dataset = "/media/vitor/SHARE/DEV/Visão Computacional/dataset/lfw_eleicao_test1" #pasta contendo a pasta de cada pessoa
all_people = os.listdir(direc_dataset)

features_face2 = np.reshape(features_face2,(495,2048))

####################
# Array com o endereço de todas as fotos do dataset
all_photo_array = []

for i in range(len(all_people)): # define quantas pastas serão percorridas

    name_person = all_people[i] # a lista name_person recebe como parametro o nome correspondente na lista all_people
            
    # Caminho da pasta de cada pessoa
    path_person_list = os.listdir(direc_dataset+"/"+name_person)
    all_photo_array.append(path_person_list)

# print(all_photo_array)
# print(len(all_photo_array))

# ######################
# print(features_face2.shape)
kmean = KMeans(n_clusters=21)
kmean.fit(features_face2)

y = kmean.predict(features_face2)

print(y)
print(len(y))
print(array_kmeans)


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

print(size_by_name)
number = 0

array_wrong_class = []


array_all_class = []
current_class = []
# Esse laço separa a análise por pessoa
for i in range(len(size_by_name)):
    for f in range(size_by_name[i]):
        # print(f)
        print(f+1, '--', y[f+number], '--', f+number)

        array_all_class.append(y[f+number])
    
    # Aqui descobrimos qual classe representa a pessoa e adicionamos em uma lista
    counts = np.bincount(array_all_class)
    current_class.append(np.argmax(counts))
    
    print('Classe da pessoa', current_class[i])
    
    print('print array_all_class', array_all_class)
    # Agora verificamos o array da pessoa em busca uma classe que não a descreve e pegar a posição dessa foto ruim
    for posicao in range(len(array_all_class)):
        if (array_all_class[posicao] != current_class[i]):

            # print('array_all_class[posicao]',array_all_class[posicao])
            # print('current_class[posicao]', current_class[i])
            # Aqui vamos acessar a pasta da pessoa e mostrar a foto que não a representa
            number_name = posicao
            name = list_label[i]
            mostrar_img(posicao, name)


    
    
    del(array_all_class[:])
    
    print("------------", f+1)
    number += (f+1) #pula pra próxima posição
    



# Data = {'x' : y, 'y' : array_kmeans}

# df = DataFrame(Data,columns=['x','y'])
# pandas.set_option('display.max_rows', df.shape[0]+1) #mostra todas as linhas
# print(df)


# # # Delete images 


# label = os.listdir(direc_dataset)
# size_by_name = []
# print(label)

# name = []
# size_by_name.append(0)
# # Esse laço retorna um vetor onde cada posição tem o número de fotos da pasta correspondente na mesma posição
# for i in range(len(label)):
#     name = label[i]
#     print(name)
#     direct_name = os.listdir(direc_dataset+'/'+name)
#     size_by_name.append(len(direct_name)+size_by_name[i])

# # deletetamos o primeiro elemento de size_by_name
# del(size_by_name[0])
# print(size_by_name)

# for i in range(len(y)):
#     for f in 



# # last_check = 0
# # dfs = []

# # for ind in size_by_name:
# #     dfs.append(y.loc[last_check:ind-1])
# #     last_check = ind
# #     # print(last_check)

# # print("===================")

# # # # Mostra todas as classes e o respective indice da pessoa
# # print(dfs)
# # print((len(dfs)))

# # # Para mostrar as classes de uma pessoas, é só printar dfs na posição correspondente
# # print(dfs[10])
# # print(len(dfs[10]))


# # # os.remove(file) for file in os.listdir('path/to/directory') if file.endswith('.png')

# # # for i in range(len(y)):
# # #     print(y[i]+' ------ '+array_kmeans[i])

# # # for files in sorted(all_people):
# # #     one_person = os.listdir(direc_dataset+'/'+files)
# # #     # print(one_person)
# # #     print(files)
# # #     for i in range(len(one_person)):
# # #         array_label.append(files)
# # #         print(y[i])