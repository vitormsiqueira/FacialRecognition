# Experimento 1
# Todo:
# 1 - VGGFace da mesma pessoa
# 2 - VGGFace de pessoas diferentes
# 3 - VGGFace é melhor que VGG16?
# 4 - Qual saida de VGGFace é melhor??

from keras_vggface.vggface import VGGFace

from keras.layers import Flatten
from keras.models import Model, Sequential
import numpy as np
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from scipy.spatial import distance
model = Sequential()

def ler_imagem(nome):
    img = image.load_img(nome, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)
    return x

def euclideanDistance(base, teste):
    var = base - teste
    var = np.sum(np.multiply(var, var))
    var = np.sqrt(var)
    return var

vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
vgg_model.summary()
model = Model(inputs=vgg_model.layers[0].input, outputs=vgg_model.layers[-2].output)
#last_layer = vgg_model.get_layer('pool5').output
#out = Flatten(name='flatten')(last_layer)

def same_person(name):
    preds = model.predict(ler_imagem('./'+name+'.png'))
    preds2 = model.predict(ler_imagem('./'+name+'2.png'))

    distancia = euclideanDistance(preds, preds2)
    print(distancia)

def dif_person():
    preds = model.predict(ler_imagem('./vitor.png'))
    preds2 = model.predict(ler_imagem('./vitor3.png'))

    distancia = euclideanDistance(preds, preds2)
    print(distancia)

same_person('vitor')
same_person('alvaro')
dif_person()

'''
vgg_model2 = VGGFace(include_top=False, input_shape=(224, 224, 3))
vgg_model2.summary()
preds2 = vgg_model2.predict(x)

print(preds.shape)
print(preds2.shape)
'''
'''
 # or version=2
preds = model.predict(x)
print('Predicted:', utils.decode_predictions(preds))
'''
