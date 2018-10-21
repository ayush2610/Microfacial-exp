from keras.utils import plot_model
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import MaxPooling3D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation

class C3D:
    def build(numSamples,height,width,channels,classes,activation='relu',weightPath=None):
        model=Sequential()

        model.add(Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',activation='relu',input_shape=(numSamples,height,width,channels)))
        model.add(MaxPooling3D((1,2,2),strides=(1,2,2),padding='valid'))

        model.add(Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',activation='relu'))
        model.add(MaxPooling3D((2,2,2),strides=(2,2,2),padding='valid'))

        model.add(Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',activation='relu'))
        model.add(Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',activation='relu'))
        model.add(MaxPooling3D((2,2,2),strides=(2,2,2),padding='valid'))

        model.add(Conv3D(512,(3,3,3),strides=(1,1,1),padding='same',activation='relu'))
        model.add(Conv3D(512,(3,3,3),strides=(1,1,1),padding='same',activation='relu',))
        model.add(MaxPooling3D((2,2,2),strides=(2,2,2),padding='valid'))

        model.add(Conv3D(512,(3,3,3),strides=(1,1,1),padding='same',activation='relu'))               
        model.add(Conv3D(512,(3,3,3),strides=(1,1,1),padding='same',activation='relu'))
        model.add(MaxPooling3D((2,2,2),strides=(2,2,2),padding='valid'))

        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        if weightPath is not None :
            model.load_weights(weightPath)
        plot_model(model,to_file="model.png", show_shapes==True,show_layer_names=True)
        return model
