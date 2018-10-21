from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.convolutional import MaxPooling3D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as keras

class VGG:
    def build(numSamples,channels,height,width,classes,activation='relu',weightPaths=None):
        model = Sequential()
        #layer 1
        model.add(Convolution3D(64,11,11,11,strides=(1,1,1),padding='valid',input_shape=(numSamples,height,width,channels)))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(3,3,3),strides=(2,2,2)))
        #layer 2
        model.add(Convolution3D(128,3,3,3,strides=(1,1,1),padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(3,3,3),strides=(2,2,2)))
        #layer 3
        model.add(Convolution3D(256,3,3,3,strides=(1,1,1),padding='valid'))
        model.add(Activation('relu'))
        #layer 4
        model.add(Convolution3D(512,3,3,3,strides=(1,1,1),padding='valid'))
        model.add(Activation('relu'))
        #layer 5
        model.add(Convolution3D(512,3,3,3,strides=(1,1,1),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(3,3,3),strides=(2,2,2)))
        #layer 6
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        #layer 7
        model.add(Dense(100))
        model.add(Activation('relu'))
        #layer 8
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        if weightPaths is not None:
            model.load_weights(weightPaths)
        return model
