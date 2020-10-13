import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pickle

mnist_data_dir='/usr/local/Anaconda/demo_1/keras_data/'
x_train=np.load(mnist_data_dir+'train_data.npy')
y_train=np.load(mnist_data_dir+'train_label.npy')

x_test=np.load(mnist_data_dir+'test_data.npy')
y_test=np.load(mnist_data_dir+'test_label.npy')

nall= x_train.shape[0]
ny, nx = 28, 28

x_test=x_test.reshape(-1, ny,nx,1)
# -1 can be nall
x_train=x_train.reshape(-1, ny,nx,1)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

no_output= 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(no_output, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=['accuracy'])

hist1 = model.fit(x_train, y_train, epochs=10, batch_size=200, verbose=1,
                  validation_data=(x_test, y_test))


model.save("model_cnn1.hdf5")

h=hist1.history
with open("history_cnn1.pkl", "wb") as f:
    pickle.dump(h, f)
    
