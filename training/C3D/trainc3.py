import c3d
from sklearn.cross_validation import train_test_split
from sklearn import datasets
import numpy as np 
from keras.optimizers import SGD                                                                                                                                                                                                                                                                                                                                        
from keras.utils import np_utils
from keras import backend as k
import cv2 
import argparse
import matplotlib.pyplot as plt
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument('-s','--save-model',type=int,default=-1,help='(optional) whether to  save the model')
ap.add_argument('-l','--load-model',type=int,default=-1,help='(optional) whether to load pretrained model')
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
ap.add_argument('-d','--dataset-path',type=str,help='path to dataset')
args = vars(ap.parse_args())

# code to load data set
path = args['dataset_path']
def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

emo_folders = os.listdir(path)
q=0
dataset={'data':[],'target':[]}
nametolabel = {'disgust':0,'fear':1,'happiness':2,'repression':3,'sadness':4,'surprise':5}
for folder in emo_folders:
    pers_folder = os.listdir(path+'/'+folder)
    for fold in pers_folder:
        imgs = os.listdir(path+'/'+folder+'/'+fold)
        imgs = sorted_aphanumeric(imgs)
        temp=[]
        for img in imgs:    
            if 're' in img:        
                temp.append(path+'/'+folder+'/'+fold+'/'+img)
        dataset['data'].append(temp)
        dataset['target'].append( nametolabel[folder])
        print(q)
        q+=1
oneset = 151
data = np.zeros((len(dataset['data']),oneset,32,32,1))
p=0
print(len(dataset['data']))
#print(len(dataset['target']))
for ie,addlis in enumerate(dataset['data']):
    #print(addlis)
    for ig ,imPath in enumerate(addlis):
        image = cv2.imread(imPath,0)
        image=cv2.resize(image,(32,32))
        print(ig)
        p+=1
        data[ie,ig,...]=image.resize((32,32,1))
#print(len(data))
print(np.shape(data[0,0]))
print(data.shape)
#print(len(dataset['target']))
traindata,testdata,trainlabels,testlabels=train_test_split(np.array(data),np.array(dataset['target']).astype(int),test_size=0.10)

trainlabels=np_utils.to_categorical(trainlabels,6)
testlabels=np_utils.to_categorical(testlabels,6)

print('Compiling Model...')
opt = SGD(lr=0.01)
model=c3d.C3D.build(numSamples=151,channels=1,height=32,width=32,activation='relu',classes=6,weightPath=args['weights'] if args['load_model']>0 else None )
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
print(args['load_model'])
if args['load_model']<0:
	print('training...')
	model.fit(traindata,trainlabels,batch_size=1,epochs=20,verbose=1)

	print('evaluating...')
	loss,accuracy=model.evaluate(testdata,testlabels,batch_size=1,verbose=1)
	print('accuracy : {:.2f}%'.format(accuracy*100))
if args['save_model']>0:
	print('saving model....')
	model.save_weights(args['weights'],overwrite=True)

for i in np.random.choice(np.arange(0,len(testlabels)),size=(6,)):
    probs=model.predict(testdata[np.newaxis,i])
    prediction = probs.argmax(axis=1)
    if k.image_data_format() == 'channel_first':
	    image=(testdata[i]).astype('uint8')
    else:
	    image = (testdata[i]).astype('uint8')
    #image = cv2.merge([image] * 3)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, str(prediction[0]), (5, 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
    np.argmax(testlabels[i])))
    plt.imshow(image)
    plt.show()




