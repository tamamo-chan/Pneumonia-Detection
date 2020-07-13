from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten
from keras import regularizers
import matplotlib.pyplot as plt
import sys
sys.path.append('./utils')
from dense import myDense
from conv2d import myConv2d
from maxpool import maxpool


batch_size = 32

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.02,
    zoom_range=0.1,
    rescale=1/225.,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'filtered/',
    target_size=(224, 224),
    class_mode='binary',
    shuffle=True,
    batch_size=batch_size,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'filtered/',
    target_size=(224, 224),
    shuffle=True,
    class_mode='binary',
    batch_size=batch_size,
    subset='validation'
)

model = Sequential()

model.add(myConv2d(64, 3, input_shape=(224, 224, 3)))
model.add(maxpool((2,2)))
model.add(myConv2d(64, 3))
model.add(maxpool((2,2)))
model.add(myDense(16, kernel_regularizer=regularizers.l2(0.005)))
model.add(Flatten())
model.add(myDense(1, activation='sigmoid'))

model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

history = model.fit_generator(
                    train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    validation_data = validation_generator,
                    validation_steps=validation_generator.samples // batch_size,
                    epochs=20)

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax = ax.ravel()

model.save('model.h5')

for i, met in enumerate(['accuracy', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])

plt.show()