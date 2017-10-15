import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
import family_generator as fg
from keras.utils.np_utils import to_categorical
import pdb

batch_size = 128
num_classes = 2
epochs = 800
input_shape = 29

family, fam_labels = fg.generate_dataset(10000, 0)
non_family, labels = fg.generate_dataset(10000, 1)

x = np.vstack((family, non_family))
y = np.hstack((fam_labels, labels))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

y_test = y_test.reshape((-1, 1))
y_train = y_train.reshape((-1, 1))
y_valid = y_valid.reshape((-1, 1))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_valid = x_valid.astype('float32')

# pdb.set_trace()
model = Sequential()
model.add(Dense(64, input_shape=(29,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,  activation='sigmoid'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_valid, y_valid))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

