import tensorflow as tf
import numpy as np
import time
import datetime
import pandas
import cv2
import os
import re
import Alexnet
import matplotlib.pyplot as plt

def get_nextbatchImage_withoutDict(image_dir,image_names,n_classes,key_classes,step,batch):
    images_batch=[]
    labels_batch=[]
    image_names_len=len(image_names)
    index_start=(step*batch)%image_names_len
    for i in range(batch):
        filename=image_names[(i+index_start)%image_names_len]
        if re.search('[jpg|jpeg|JPG|JPEG|png|PNG]',filename):
            image_path=os.path.join(image_dir,filename)
            image_tmp=cv2.imread(image_path)
            image_tmp=image_tmp*(1.0/255)-0.5
            image_ret=image_tmp.astype(np.float32)
            label_onehot=np.zeros(n_classes,dtype=np.float32)
            label_name_index=key_classes.index(filename.split('/')[0])
            label_onehot[label_name_index]=1
            images_batch.append(image_ret)
            labels_batch.append(label_onehot)    
    return images_batch,labels_batch 

def get_RandomTestingImage_withoutDict(image_dir,image_names,n_classes,key_classes,random_batch):
    images_batch=[]
    labels_batch=[]
    image_names_len=len(image_names)
    for i in range(random_batch):
        filename=image_names[np.random.randint(image_names_len)]
        if re.search('[jpg|jpeg|JPG|JPEG|png|PNG]',filename):
            image_path=os.path.join(image_dir,filename)
            image_tmp=cv2.imread(image_path)
            image_tmp=image_tmp*(1.0/255)-0.5
            image_ret=image_tmp.astype(np.float32)
            label_onehot=np.zeros(n_classes,dtype=np.float32)
            label_name_index=key_classes.index(filename.split('/')[0])
            label_onehot[label_name_index]=1
            images_batch.append(image_ret)
            labels_batch.append(label_onehot)    
    return images_batch,labels_batch 

def get_AllLabel(image_dir,LabelTxtName):
    txt_path=os.path.join(image_dir,LabelTxtName)
    image_names=[]
    if not os.path.exists(txt_path):
        print 'func get_nextbatchImage:err.please input correct image_dir'
        return     
    with open(txt_path,'r') as f:
        for line in f:
            image_names.append(line.replace('\n',''))    
    return image_names

random_batch=100
test_t0=time.time()
INPUT_DATA='./data/mydata'
testingLabelTxt='testing_labels.txt'
category_txt='sample_category.txt'
testing_image_names=get_AllLabel(INPUT_DATA,testingLabelTxt)
key_classes=get_AllLabel(INPUT_DATA,category_txt)
n_classes=len(key_classes)
Testimages_batch,Testlabels_batch=get_RandomTestingImage_withoutDict(INPUT_DATA,testing_image_names,n_classes,key_classes,random_batch)

with tf.Session() as sess:
    saver=tf.train.import_meta_graph('./model/mymodel/AlexnetModel_2999.meta')
    saver.restore(sess,'./model/mymodel/AlexnetModel_2999')
    graph=tf.get_default_graph()
    images_input=graph.get_tensor_by_name("images_input:0")
    labels_input=graph.get_tensor_by_name("labels_input:0")
    evalution=graph.get_tensor_by_name("evalution:0")
    feed_dict={images_input:Testimages_batch,labels_input:Testlabels_batch}
    print '===================== Start Testing ====================>'
    with tf.device("/gpu:0"):
        test_evalution_np=sess.run(evalution,feed_dict)
    print ('Testing on random sample:%d =====>Accuracy:%0.2f%%'%(random_batch,test_evalution_np*100))
    print '===================== Finish Testing ====================>'
    print 'Testing Time is : %0.2fs\n'%(time.time()-test_t0)