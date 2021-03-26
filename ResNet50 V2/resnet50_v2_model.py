from keras.applications.resnet_v2 import ResNet50V2
model=ResNet50V2(include_top=True, weights=None, input_tensor=None, input_shape=(100,100,3),classes=41)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('Compiled!')

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
batch_size = 50

checkpointer = ModelCheckpoint(filepath = 'cnn_from_scratch_fruits.hdf5',  save_best_only = True)

history = model.fit(x_train,y_train,
        batch_size = 50,
        epochs=15,
        validation_data=(x_valid, y_vaild),
        callbacks = [checkpointer],
                    shuffle=True
        )
        
model.load_weights('cnn_from_scratch_fruits.hdf5')

score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
