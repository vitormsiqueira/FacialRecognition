
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from keras.models import Model, Sequential
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os

vggface2_model = VGGFace(model='resnet50', include_top=True)
vggface2_model = Model(inputs=vggface2_model.layers[0].input, outputs=vggface2_model.layers[-3].output)
vggface2_model.summary()

direc_dataset = "/media/vitor/SHARE/DEV/Visão Computacional/dataset/lfw_eleicao_test1/" #pasta contendo a pasta de cada pessoa
all_persons = os.listdir(direc_dataset)

array_kmeans = []
array_preds = []

def extrair(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    preds = vggface2_model.predict(img_data)
    array_preds.append(preds)

    return array_preds

for i in sorted(range(len(all_persons))):
    person_path = direc_dataset+'/'+all_persons[i]
    current_person = os.listdir(person_path)
    for f in sorted(current_person):
        array_kmeans.append(i)
        print(array_kmeans)
        print(len(array_kmeans))
        img_path = person_path+'/'+f
        extrair(img_path)
        data_face2_features = np.asarray(array_preds)

data_kmeans_features = np.asarray(array_kmeans)
print(array_kmeans)
print(len(array_kmeans))
np.save('/media/vitor/SHARE/DEV/Visão Computacional/siamese_net/kmeans/004data_face2_features', data_face2_features) # salva os arrays de distancia
np.save('/media/vitor/SHARE/DEV/Visão Computacional/siamese_net/kmeans/004data_kmeans', data_face2_features) # salva os arrays de distancia

