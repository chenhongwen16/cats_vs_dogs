import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D,GlobalMaxPooling2D


batch_size = 32
num_classes = 10
epochs = 1600
data_augmentation = True

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print('x_train shape:',x_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /=255

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

model = Sequential()

model.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3),padding='same',input_shape=x_train[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3),padding='same',input_shape=x_train[1:]))
model.add(Activation('relu'))
model.add(Conv2D(48,(3,3),padding='same',input_shape=x_train[1:]))
model.add(Activation('relu'))
model.add(Conv2D(48,(3,3),padding='same',input_shape=x_train[1:]))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(80,(3,3),padding='same',input_shape=x_train[1:]))
model.add(Activation('relu'))
model.add(Conv2D(80,(3,3),padding='same',input_shape=x_train[1:]))
model.add(Activation('relu'))
model.add(Conv2D(80,(3,3),padding='same',input_shape=x_train[1:]))
model.add(Activation('relu'))
model.add(Conv2D(80,(3,3),padding='same',input_shape=x_train[1:]))
model.add(Activation('relu'))
model.add(Conv2D(80,(3,3),padding='same',input_shape=x_train[1:]))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),padding='same',input_shape=x_train[1:]))
model.add(Activation('relu'))
model.add(Conv2D(128,(3,3),padding='same',input_shape=x_train[1:]))
model.add(Activation('relu'))
model.add(Conv2D(128,(3,3),padding='same',input_shape=x_train[1:]))
model.add(Activation('relu'))
model.add(Conv2D(128,(3,3),padding='same',input_shape=x_train[1:]))
model.add(Activation('relu'))
model.add(Conv2D(128,(3,3),padding='same',input_shape=x_train[1:]))
model.add(Activation('relu'))

model.add(GlobalMaxPooling2D())
model.add(Dropout(0.25))

model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

opt = keras.optimizers.Adam(lr=0.001)

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

print("train________________")
model.fit(x_train,y_train,epochs=600,batch_size=128,)
print("test_________________")
loss,acc=model.evaluate(x_test,y_test)
print("loss=",loss)
print("accuracy=",acc)