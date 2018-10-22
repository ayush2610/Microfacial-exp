import cv2 
import numpy as np 
import os 
import re
import matplotlib.pyplot as plt

path = '/home/ayush/DEEP_LEARNING/Resources/databases/casme2_updated'
newpath='/home/ayush/DEEP_LEARNING/Resources/databases/casme2_salienc'
emo_folders = os.listdir(path)
allimg = []
persfolder=[]
newpersfolder=[]

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


for folder in emo_folders:
    pers_folder = os.listdir(path+'/'+folder)
    for fold in pers_folder:
        imgs = os.listdir(path+'/'+folder+'/'+fold)
        imgs = sorted_aphanumeric(imgs)
        tempimg=[]
        for img in imgs:
            if "re" in img:
                tempimg.append(img)
        allimg.append(tempimg)
        persfolder.append(path+'/'+folder+'/'+fold)
       # newpersfolder.append(newpath+'/'+folder+'/'+fold)

data = np.zeros((151, 64,64,3))

def findnSaveOpt(index,med):
    imgAdd = persfolder[index] + '/' + allimg[index][0]
    for z,im in enumerate(allimg[index][0:]):
        frame = cv2.imread(persfolder[index] + '/' + im )
        salience = frame-med
        cv2.imshow('na',salience)
        cv2.imwrite(persfolder[index]+'/saliency'+str(z)+'.png',salience)
        print('saved Image')
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def load_pers(index):
    for nu, im in enumerate(allimg[index][0:]):
        frame = cv2.imread(persfolder[index] + '/' + im )
        data[nu,...]=frame
        print(im)
    med_img = np.median(data,axis=0)
    return med_img


for i in range(len(persfolder)):
    print(i)
    med=load_pers(i)
    findnSaveOpt(i,med)

