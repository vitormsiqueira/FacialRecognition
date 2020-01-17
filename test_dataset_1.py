from PIL import Image
from glob import glob
from random import randint
import os
from itertools import combinations

diretorio_dataset = "/media/vitor/SHARE/DEV/Visão Computacional/data/DATASET_lfw/test_2" #pasta contendo a pasta de cada pessoa

def same_person(len_by_person, one_person_list):

    print(list(combinations(one_person_list, 2))) #gera lista de combinações entre duas fotos da mesma pessoa

    print("\n")
    


def ger_pessoas(diretorio_dataset):
    all_persons = os.listdir(diretorio_dataset) # recebe o nome de cada pasta, onde se encontra cada pessoa
    one_person = []
    #print(all_persons)
    for i in range(len(all_persons)):
        one_person = all_persons[i]
        one_person_list = os.listdir("/media/vitor/SHARE/DEV/Visão Computacional/data/DATASET_lfw/test_2"+"/"+one_person)
        len_by_person = len(one_person_list)

        if len_by_person > 1:
            same_person(len_by_person, one_person_list)


ger_pessoas(diretorio_dataset)