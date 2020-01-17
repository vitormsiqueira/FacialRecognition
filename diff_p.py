from PIL import Image
from glob import glob
from random import randint
import os
import numpy as np


direc_dataset = "/media/vitor/SHARE/DEV/Visão Computacional/siamese_net/lfw_cropp" #pasta contendo a pasta de cada pessoa

def ger_pessoas(direc_dataset):
    
    all_persons = os.listdir(direc_dataset) # recebe o nome de cada pasta, onde se encontra cada pessoa
    name_person = []

    array_all_pic = [] # array contendo todas as fotos     
    
    # name_person = all_persons[0]
    # list_1 = os.listdir(direc_dataset+"/"+name_person) #lista o diretorio da pessoa

    for i in range(len(all_persons)):
        name_person = all_persons[i]
        list_1 = os.listdir(direc_dataset+"/"+name_person) #lista o diretorio da pessoa

        array_all_pic = np.concatenate((array_all_pic, list_1))
    
    print(array_all_pic)
    print(len(array_all_pic))

    print(array_all_pic[285])

    return array_all_pic

array_all_pic = ger_pessoas(direc_dataset)


from random import randint
photo_1 = randint(0,4321)
photo_2 = randint(0,4321)

print(photo_1, photo_2)

photo_1 = array_all_pic[photo_1]
photo_2 = array_all_pic[photo_2]

photo_1 = photo_1[:-9]
print(photo_1)

