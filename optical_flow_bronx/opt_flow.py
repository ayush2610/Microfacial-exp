import cv2 
import numpy as np 
import os 
import re
import matplotlib.pyplot as plt

path = '/home/ayush/DEEP_LEARNING/Resources/databases/casme2_updated'
emo_folders = os.listdir(path)
allimg = []
persfolder=[]

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


for folder in emo_folders:
    pers_folder = os.listdir(path+'/'+folder)
    for fold in pers_folder:
        imgs = os.listdir(path+'/'+folder+'/'+fold)
        imgs = sorted_aphanumeric(imgs)
        allimg.append(imgs)
        persfolder.append(path+'/'+folder+'/'+fold)

print(persfolder[111])
sum=0

def findnSaveOpt(index):
    imgAdd = persfolder[index] + '/' + allimg[index][0]
    print(imgAdd)
    frame1 = cv2.imread(imgAdd)
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    z=0
    for im in allimg[index][1:]:
        frame2 = cv2.imread(persfolder[index] + '/' + im ) 
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        #next=cv2.GaussianBlur(next,(3,3),0)
        #prvs=cv2.GaussianBlur(prvs,(3,3),0)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 1,32, 5, 8, 2.0, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        #plt.imshow(bgr)
        #plt.show()
        #plt.pause(0.1)
        z+=1
        cv2.imwrite(persfolder[index]+'/optflowavg-'+str(z)+'.png',bgr)
        print('saved Image')
        prvs = next

for i in range(len(persfolder)):
    print(i)
    findnSaveOpt(i)

