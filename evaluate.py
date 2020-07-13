from keras.models import load_model
import sys

from keras.models import Sequential
from keras.models import load_model

sys.path.append('./utils')
from dense import myDense
from conv2d import myConv2d
from maxpool import maxpool
from sequence import DataGenerator
from os import listdir

from pycm import ConfusionMatrix

model = Sequential()

filelist = []
labels = []

for file in listdir('./encoded/NORMAL'):
    filelist.append('./encoded/NORMAL/{}'.format(file))
    labels.append(0)

for file in listdir('./encoded/PNEUMONIA'):
    filelist.append('./encoded/PNEUMONIA/{}'.format(file))
    labels.append(1)

generator = DataGenerator(filelist, labels)

model = load_model('./model.h5', custom_objects={'myDense':myDense, 'myConv2d':myConv2d, 'maxpool':maxpool})

probs = model.evaluate_generator(generator)

print("The model had an accuracy of {}% with a loss of {}.".format(probs[1]*100, probs[0]))

yhat_probs = model.predict_generator(generator, verbose=0)

yhat_probs = yhat_probs[:, 0]

yhat_probs = yhat_probs.round().astype(int)

cm = ConfusionMatrix(labels, yhat_probs)
print(cm)
