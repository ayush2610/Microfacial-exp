import keras
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense, Input, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D

class NewNet:
    def build(numSamples,height,width,channels,classes,weightPath, activation='relu'):
        #input 1
        In1 = Input(shape=(numSamples,height,width,channels))
        x1  = Conv3D(8,(3,3,3),strides=(2,2,2),padding ='valid', activation='relu')(In1)

        #input 2
        In2 = Input(shape=(numSamples,height,width,channels))
        x2  = Conv3D(8,(3,3,3),strides=(2,2,2),padding ='valid', activation='relu')(In2)

        #concat
        x  = keras.layers.concatenate([x1,x2])

        xn1 = Conv3D(16,(3,3,3),strides=(2,2,2),padding ='valid', activation='relu')(xn1)
        xn2 = Conv3D(16,(5,5,5),strides=(2,2,2),padding ='valid', activation='relu')(xn2)
        xn3 = Conv3D(16,(7,7,7),strides=(2,2,2),padding ='valid', activation='relu')(xn3)
        xn4 = Conv3D(16,(11,11,11),strides=(2,2,2),padding ='valid', activation='relu')(xn4)

        #layers 2

        xn1 = Conv3D(32,(3,3,3),strides=(2,2,2),padding ='valid', activation='relu')(xn1)
        xn2 = Conv3D(32,(3,3,3),strides=(2,2,2),padding ='valid', activation='relu')(xn2)
        xn3 = Conv3D(32,(5,5,5),strides=(2,2,2),padding ='valid', activation='relu')(xn3)
        xn4 = Conv3D(32,(7,7,7),strides=(2,2,2),padding ='valid', activation='relu')(xn4)

        #layers 3


        xn1 = Conv3D(64,(3,3,3),strides=(2,2,2),padding ='valid', activation='relu')(xn1)
        xn2 = Conv3D(64,(3,3,3),strides=(2,2,2),padding ='valid', activation='relu')(xn2)
        xn3 = Conv3D(64,(3,3,3),strides=(2,2,2),padding ='valid', activation='relu')(xn3)
        xn4 = Conv3D(64,(5,5,5),strides=(2,2,2),padding ='valid', activation='relu')(xn4)

        #layers 4

        xn1 = Conv3D(96,(3,3,3),strides=(2,2,2),padding ='valid', activation='relu')(xn1)
        xn2 = Conv3D(96,(3,3,3),strides=(2,2,2),padding ='valid', activation='relu')(xn2)
        xn3 = Conv3D(96,(3,3,3),strides=(2,2,2),padding ='valid', activation='relu')(xn3)
        xn4 = Conv3D(96,(3,3,3),strides=(2,2,2),padding ='valid', activation='relu')(xn4)

        x  = keras.layers.concatenate([xn1,xn2,xn3,xn4])

        x = Conv3D(128,(3,3,3),strides=(2,2,2),padding ='valid', activation='relu')(x)
        x  = Dense(classes)(x)
        x  = Activation('softmax')(x)
        
        if weightPath is not None:
            model.load_weights(weightPath)
        plot_model(model,to_file="model.png", show_shapes=True,show_layer_names=True)
        return model


        
