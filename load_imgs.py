import numpy as np
import os

# Pasta contendo a pasta de cada pessoa
direc_dataset = "/media/vitor/SHARE/DEV/Visão Computacional/dataset/lfw_eleicao_test1" 

# Recebe o nome de cada pasta, onde se encontra cada pessoa
all_people = os.listdir(direc_dataset) 
# print('+++++++',all_people)
#cria uma lista onde iremos armazenar o nome de cada pessoa (nome da pasta)
name_person = [] 


# Array com o endereço de todas as fotos do dataset
all_photo_array = []

for i in range(len(all_people)): # define quantas pastas serão percorridas

    name_person = all_people[i] # a lista name_person recebe como parametro o nome correspondente na lista all_people
        
    # Caminho da pasta de cada pessoa
    path_person_list = os.listdir(direc_dataset+"/"+name_person)

    all_photo_array.append(path_person_list)


array_all_photos = []

    # Agora vamos acessar a imagem

pessoa = direc_dataset+'/'+all_people[0]
    
    # Listamos todas as fotos e armazenamos em um array
    
array_all_photos.append(path_person_list)

print('222222', array_all_photos)

print(array_all_photos[5])

print(all_photo_array)
print(len(all_photo_array))

