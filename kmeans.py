import numpy as np
import os
from keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from keras.models import Model, Sequential
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans


vggface2_model = VGGFace(model='resnet50', include_top=True)
# vggface2_model.summary()
vggface2_model = Model(inputs=vggface2_model.layers[0].input, outputs=vggface2_model.layers[-3].output)

features_face2 = np.load('/media/vitor/SHARE/DEV/Visão Computacional/siamese_net/kmeans/004data_face2_features.npy') # load
array_kmeans = np.load('/media/vitor/SHARE/DEV/Visão Computacional/siamese_net/kmeans/004data_kmeans.npy') # load

print(features_face2.shape)

direc_dataset = "/media/vitor/SHARE/DEV/Visão Computacional/dataset/lfw_eleicao_test1/" #pasta contendo a pasta de cada pessoa
label = os.listdir(direc_dataset)



array_final = []

for i in range(501):
    array_final.append(features_face2[i])

# print(array_final)
# print(len(array_final))
# print(type(features_face2))

direc_dataset = "/media/vitor/SHARE/DEV/Visão Computacional/dataset/lfw_cropp" #pasta contendo a pasta de cada pessoa
all_persons = os.listdir(direc_dataset)

features_face2 = np.reshape(features_face2,(501,2048))

print(features_face2.shape)
kmean = KMeans(n_clusters=21)
kmean.fit(features_face2)

y = kmean.predict(features_face2)

print(kmean)


for file in sorted(all_persons):
    one_person = os.listdir(direc_dataset+'/'+file)
    for qtd in sorted(one_person):
        for i in y:
            print(i)
        print(len(y))

