#booster.utility  calc_forward_flows_brox load_forward_flows load_backward_flows 
#booster.booster  interpolate_frames_occlusion
#booster.flo
import numpy as np
import cv2
import pickle
import glob, os
import tools.utility as ut
import tools.booster as bst
import tools.color_transfer as ct
import tools.flo as flo
import re
import tools.pixels as pix
import os.path

path = '/home/ayush/DEEP_LEARNING/Resources/databases/casme2_updated'
def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

emo_folders = os.listdir(path)
q=0
targetfrms=150
dataset={'data':[],'target':[]}
paths=[]
nametolabel = {'disgust':0,'fear':1,'happiness':2,'repression':3,'sadness':4,'surprise':5}
for folder in emo_folders:
    pers_folder = os.listdir(path+'/'+folder)
    for fold in pers_folder:
        imgs = os.listdir(path+'/'+folder+'/'+fold)
        imgs = sorted_aphanumeric(imgs)
        temp=[]
        tpath=[]
        for img in imgs:            
            if len(img) <= 7:
                temp.append(path+'/'+folder+'/'+fold+'/'+img)
                tpath.append(path+'/'+folder+'/'+fold+'/')
                #print(folder)
        dataset['data'].append(temp)
        paths.append(tpath)
        dataset['target'].append( nametolabel[folder])
        print(q)
        q+=1
oneset = 150
data = []
p=0
print('lo',len(dataset['data']))
#print(len(dataset['target']))
for ie,addlis in enumerate(dataset['data']):
    #print(addlis)
    temp=[]
    for ig ,imPath in enumerate(addlis):
        image = cv2.imread(imPath)
        image=cv2.resize(image,(150,150))
        print(p)
        p+=1
        temp.append(image)
    data.append(temp)

print(dataset['data'][0][0])
print(paths[0][0])
#cv2.waitKey(0)
n=0
for re in range(len(data)):
    nfrs = len(data[re])-1
    mod = targetfrms%nfrs
    div = targetfrms/nfrs
    total=0
    print(mod,div,'kk')
    count=1
    cv2.imwrite(paths[re][0]+'normalised'+str(count)+'.jpg',data[re][0])
    print('Saving: '+paths[re][0]+'normalised'+str(count)+'.jpg')
 
    for xe in range(0,len(data[re])-1):
        #cv2.imshow('lo',data[0])
        #cv2.waitKey(0)
        print(re,xe)
        z=mod
        if mod>5:
            z=5
        mod=mod-z
        n=z+int(div)-1
        count+=1
        cv2.imwrite(paths[re][xe]+'normalised'+str(count)+'.jpg',data[re][xe])
        print('Saving: '+paths[re][xe]+'normalised'+str(count)+'.jpg')

        
        for ij in range(1,n+1):  
            count+=1      
            if not os.path.exists(paths[re][xe]+'normalised'+str(count)+'.jpg'):
                t=max(0.0,(min(1.0,ij/(n+1))))
                forward=ut.optical_flow(data[re][xe],data[re][xe+1])
                backward=ut.optical_flow(data[re][xe+1],data[re][xe])
                flow1 = pix.splat_motions_bidi(forward, backward,
                                     data[re][xe], data[re][xe+1], t)
                flow1 = cv2.GaussianBlur(flow1, (11, 11), 10)
                interpolated1 = ct.color_transfer_occlusions(data[re][xe],
                                                    data[re][xe+1],
                                                    forward,
                                                    backward,
                                                    flow1,
                                                    t)
            #cv2.imshow('inter'+str(ij),interpolated1)
            #cv2.imshow('image1',data[re][xe])
            #cv2.imshow('image2',data[re][xe+1])
            
                #count+=1
                print('Saving: '+paths[re][xe]+'normalised'+str(count)+'.jpg')
                cv2.imwrite(paths[re][xe]+'normalised'+str(count)+'.jpg',interpolated1)
        
    
        #cv2.imshow('optflow',forward)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        total+=n
    count+=1
    print('Saving: '+paths[re][-1]+'normalised'+str(count)+'.jpg')
    cv2.imwrite(paths[re][-1]+'normalised'+str(count)+'.jpg',data[re][-1])
    print('totalframes',total+nfrs)
