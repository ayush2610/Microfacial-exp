import newNetwork
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np 
from keras.optimizers import SGD #GD                                                                                                                                                                                                                                                                                                                                        
from keras.utils import np_utils
from keras import backend as k
import cv2 
import argparse
import matplotlib.pyplot as plt
import os
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config=tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement=True
sess = tf.Session(config=config)
set_session(sess)

ap = argparse.ArgumentParser()
ap.add_argument('-s','--save-model',type=int,default=-1,help='(optional) whether to  save the model')
ap.add_argument('-l','--load-model',type=int,default=-1,help='(optional) whether to load pretrained model')
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
ap.add_argument('-d1','--dataset1-path',type=str,help='path to dataset real')
ap.add_argument('-d2', '--dataset2-path',type=str,help='path to dataset saliency')
args = vars(ap.parse_args())

# code to load data set
path1 = args['dataset1_path']
path2 = args['dataset2_path']
def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

emo_folders = os.listdir(path1)
q=0


dataset={'data':[],'target':[]}
nametolabel = {'disgust':0,'fear':1,'happiness':2,'repression':3,'sadness':4,'surprise':5}
for folder in emo_folders:
    pers_folder = os.listdir(path1+'/'+folder)
    for fold in pers_folder:
        imgs = os.listdir(path1+'/'+folder+'/'+fold)
        imgs = sorted_aphanumeric(imgs)
        temp=[[]]*28

        countmod=0
        for img in imgs:    
            if 'norm' in img:
                temp[countmod]=np.append(temp[countmod],(path1+'/'+folder+'/'+fold+'/'+img))
                countmod=(countmod+1)%28
        #print(len(temp[8]))
        dataset['data'].extend(temp)
        dataset['target'].extend( [nametolabel[folder]]*28)
        print(q)
        q+=1
oneset = 152
data = np.zeros((len(dataset['data']),oneset,112,112,3),dtype=np.uint8)
p=0
print(len(dataset['data']))
#print(len(dataset['target']))
for ie,addlis in enumerate(dataset['data']):
    #print(addlis)
    for ig ,imPath in enumerate(addlis):
        image = cv2.imread(imPath)
        image=cv2.resize(image,(112,112))
        print(ig,imPath)
        p+=1
        data[ie,ig,...]=image
#print(len(data))


emo_folders = os.listdir(path2)
q=0
dataset2={'data':[],'target':[]}
for folder in emo_folders:
    pers_folder = os.listdir(path2+'/'+folder)
    for fold in pers_folder:
        imgs = os.listdir(path2+'/'+folder+'/'+fold)
        imgs = sorted_aphanumeric(imgs)
        temp=[[]]*28
        countmod=0

        for img in imgs:    
            temp[countmod]=np.append(temp[countmod],(path2+'/'+folder+'/'+fold+'/'+img))
            countmod=(countmod+1)%28
        dataset2['data'].extend(temp)
        dataset2['target'].extend( [nametolabel[folder] ]*28)
        print(q)
        q+=1
oneset = 152
data2 = np.zeros((len(dataset2['data']),oneset,112,112,3),dtype=np.uint8)
p=0
print(len(dataset2['data']))
#print(len(dataset['target']))
for ie,addlis in enumerate(dataset2['data']):
    #print(addlis)
    for ig ,imPath in enumerate(addlis):
        image = cv2.imread(imPath)
        image=cv2.resize(image,(112,112))
        print(ig,imPath)
        p+=1
        data2[ie,ig,...]=image

print(np.shape(data2[0,0]))
print(data2.shape)
del dataset['data']
del dataset2['data']
#print(len(dataset['target']))
data_f = np.stack((data,data2),axis=-1)
del data
del data2
print(1)
traindata,testdata,trainlabels,testlabels=train_test_split(np.array(data_f),np.array(dataset['target']).astype(int),test_size=0.10)
print(2)
del data_f
trainlabels=np_utils.to_categorical(trainlabels,6)
testlabels=np_utils.to_categorical(testlabels,6)
print(3)
finalTrainData = np.split(traindata,2,axis=-1)
print(4)
finaltestdata = np.array(np.split(testdata,2,axis=-1))
del traindata
del testdata
print('Compiling Model...')
opt = SGD(lr=0.0001)
model=newNetwork.NewNet.build(numSamples=152,width=112,height=112,channels=3,activation='relu',classes=6,weightPath=args['weights'] if args['load_model']>0 else None )
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
print(args['load_model'])
if args['load_model']<0:
	print('training...')
	history=model.fit(finalTrainData,trainlabels,batch_size=1,epochs=200,verbose=1)
	# Plot training & validation accuracy values
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
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
if args['save_model']>0:
    print('saving model....')
    model.save_weights(args['weights'],overwrite=True)

print('evaluating...')
loss,accuracy=model.evaluate([finaltestdata[0],finaltestdata[1]],testlabels,batch_size=1,verbose=1)
print('accuracy : {:.2f}%'.format(accuracy*100))


for i in range(15):
    #print(finaltestdata.shape)
    probs=model.predict([finaltestdata[0][i][np.newaxis,:],finaltestdata[1][i][np.newaxis,:]])
    prediction = probs.argmax(axis=1)
    print('prediction',(prediction))
    if k.image_data_format() == 'channel_first':
	    image=(finaltestdata[0][i][0]).astype('uint8')
    else:
	    image = (finaltestdata[0][i][0]).astype('uint8')
    #image = cv2.merge([image] * 3)
    image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, str(prediction[0]), (5, 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
    np.argmax(testlabels[i])))
    cv2.imshow('nam',image)
    cv2.waitKey(1000)




