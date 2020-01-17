import os

all_persons = os.listdir("/media/vitor/SHARE/DEV/Visão Computacional/label/images") # recebe o nome de cada pasta, onde se encontra cada pessoa
print(all_persons[0])
one_person = []
len_list = len(all_persons)
print(len_list)

def create_dic(all_persons, len_list):
    dic_persons = {
    }
    
    for i in range(len_list):
        p = all_persons[i]
        print(p)
        dic_persons[p]: i
        print(dic_persons)
    
    return dic_persons


"""
for i in range(len(all_persons)):
    one_person = all_persons[i]
    print (one_person, len(all_persons))
"""