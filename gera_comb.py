from PIL import Image
from glob import glob
from random import randint
import os
from itertools import combinations

direc_dataset = "/media/vitor/SHARE/DEV/Visão Computacional/data/DATASET_lfw/test_2/" #pasta contendo a pasta de cada pessoa

all_persons = os.listdir(direc_dataset) # recebe o nome de cada pasta, onde se encontra cada pessoa
name_person = [] #cria uma lista onde iremos armazenar o nome de cada pessoa (nome da pasta)
name_person = all_persons[0] # a lista name_person recebe como parametro o nome correspondente na lista all_persons

name_person_list = os.listdir("/media/vitor/SHARE/DEV/Visão Computacional/data/DATASET_lfw/test_2"+"/"+name_person) #lista o diretorio da pessoa


direct_1 = direc_dataset+"/"+name_person

photo1 = name_person_list[0]

print(photo1)