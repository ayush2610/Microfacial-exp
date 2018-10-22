import keras
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense, Input, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D


class twoStream:
    def build(numSamples,height,width,channels,classes,weightPath, activation='relu'):
        #stream 1
        In1 = Input(shape=(numSamples,height,width,channels))
        x1  = Conv3D(128,(3,3,3),strides=(1,1,1),padding ='same', activation='relu')(In1)
        x1  = MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2),padding = 'valid')(x1)

        x1  = Conv3D(256,(3,3,3),strides=(1,1,1),padding ='same', activation='relu')(x1)
        x1  = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),padding = 'valid')(x1)
                
        #x1  = Conv3D(256,(3,3,3),strides=(1,1,1),padding ='same', activation='relu')(x1)
        #x1  = Conv3D(256,(3,3,3),strides=(1,1,1),padding ='same', activation='relu')(x1)
        #x1  = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),padding = 'valid')(x1)

        #stream 2
        In2 = Input(shape=(numSamples,height,width,channels))
        x2  = Conv3D(128,(3,3,3),strides=(1,1,1),padding= 'same', activation='relu')(In2)
        x2  = MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2),padding = 'valid')(x2)

        x2  = Conv3D(256,(3,3,3),strides=(1,1,1),padding ='same', activation='relu')(x2)
        x2  = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),padding = 'valid')(x2)
                
        #x2  = Conv3D(256,(3,3,3),strides=(1,1,1),padding ='same', activation='relu')(x2)
        #x2  = Conv3D(256,(3,3,3),strides=(1,1,1),padding ='same', activation='relu')(x2)
        #x2  = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),padding = 'valid')(x2)

        #concatenated
        x  = keras.layers.concatenate([x1,x2])
        x  = Conv3D(512,(3,3,3),strides=(1,1,1),padding ='same', activation='relu')(x)
        x  = Conv3D(512,(3,3,3),strides=(1,1,1),padding ='same', activation='relu')(x)
        #x  = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),padding = 'valid')(x)

        x  = Conv3D(512,(3,3,3),strides=(1,1,1),padding ='same', activation='relu')(x)
        #x  = Conv3D(512,(3,3,3),strides=(1,1,1),padding ='same', activation='relu')(x)
        x  = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),padding = 'valid')(x)

        x  = Flatten()(x)
        x  = Dense(100)(x)
        x  = Activation('relu')(x)

        x  = Dense(100)(x)
        x  = Activation('relu')(x)

        x  = Dense(classes)(x)
        x  = Activation('softmax')(x)

        model = Model(inputs = [In1,In2], outputs = x)
        if weightPath is not None:
            model.load_weights(weightPath)
        plot_model(model,to_file="model.png", show_shapes=True,show_layer_names=True)
        return model
    

    

        
