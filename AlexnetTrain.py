import numpy as np 
import os
import shutil
import tensorflow as tf
import time
from tensorflow.python.platform import gfile
import re
import Alexnet
import cv2
import pickle
import datetime

MODELSAVE_DIR='./model/mymodel1'
INPUT_DATA='./data/mydata'
trainingLabelTxt='training_labels1.txt'
validationLabelTxt='validation_labels1.txt'
testingLabelTxt='testing_labels1.txt'

trainingLog='./LogFile/training_log1.log'
validationLog='./LogFile/validation_log1.log'
testingLog='./LogFile/testing_log1.log'

category_txt='sample_category1.txt'

VALIDATION_PERCENTAGE=10
TEST_PERCENTAGE=10

STEPS=4000
BATCH=100

LEARNING_RATE=0.1


def create_image_list(testing_percentage,validation_percentage):
    result={}
    sub_dirs=[os.path.join(INPUT_DATA,filename) for filename in os.listdir(INPUT_DATA)]
    for sub_dir in sub_dirs:
        dir_name=os.path.basename(sub_dir)
        label_name=dir_name.lower()
        file_list=[]
        if not os.path.isdir(sub_dir):
            continue
        if len(os.listdir(sub_dir))<10:
            continue
        for filename in os.listdir(sub_dir):
            if re.search('[jpg|jpeg|JPG|JPEG|png|PNG]',filename):
                file_list.append(filename)
        training_images=[]
        testing_images=[]
        validation_images=[]
        file_list_len=len(file_list)
        index_random_list=np.arange(file_list_len)
        np.random.shuffle(index_random_list)
        step1=int(testing_percentage*file_list_len/100)
        step2=int((testing_percentage+validation_percentage)*file_list_len/100)
        for i in range(file_list_len):
            if i<step1:
                testing_images.append(file_list[i])
            elif i>=step1 and i<step2:
                validation_images.append(file_list[i])
            else:
                training_images.append(file_list[i])
        result[label_name]={'dir':dir_name,
                            'training':training_images,
                            'testing':testing_images,
                            'validation':validation_images}
    return result

def get_image_path(image_lists,image_dir,label_name,index,category):
    label_lists=image_lists[label_name]
    category_list=label_lists[category]
    mod_index=index%len(category_list)
    basename=category_list[mod_index]
    sub_dir=label_lists['dir']
    full_path=os.path.join(image_dir,sub_dir,basename)
    return full_path


def WriteImageLabelTxt(image_lists,image_dir):
    if not os.path.exists(image_dir):
        print 'func WriteImageLabel:err.please input correct image_dir'
        return 
    if image_lists is None:
        print 'func WriteImageLabel:err.please input correct image_lists'
        return 
    training_labels=[]
    validation_labels=[]
    testing_labels=[]

    for key in image_lists.keys():
        sub_dir=image_lists[key]['dir']
        for train_imagename in image_lists[key]['training']:
            training_labels.append(os.path.join(sub_dir,train_imagename))
        for test_imagename in image_lists[key]['testing']:
            testing_labels.append(os.path.join(sub_dir,test_imagename))
        for validation_imagename in image_lists[key]['validation']:
            validation_labels.append(os.path.join(sub_dir,validation_imagename))

    training_labels_array=np.array(training_labels)
    testing_labels_array=np.array(testing_labels)
    validation_labels_array=np.array(validation_labels)

    for i in range(10):
        np.random.shuffle(training_labels_array)
        np.random.shuffle(testing_labels_array)
        np.random.shuffle(validation_labels_array)

    with open(os.path.join(image_dir,trainingLabelTxt),'w') as f:
        for filename in training_labels_array:
            f.write(filename+'\n')
    with open(os.path.join(image_dir,testingLabelTxt),'w') as f:
        for filename in testing_labels_array:
            f.write(filename+'\n')
    with open(os.path.join(image_dir,validationLabelTxt),'w') as f:
        for filename in validation_labels_array:
            f.write(filename+'\n')   

def get_nextbatchImage(imageDataDict,image_names,n_classes,key_classes,step,batch):
    images_batch=[]
    labels_batch=[]
    image_names_len=len(image_names)
    index_start=(step*batch)%image_names_len
    for i in range(batch):
        filename=image_names[(i+index_start)%image_names_len]
        if re.search('[jpg|jpeg|JPG|JPEG|png|PNG]',filename):
            image_ret=imageDataDict[filename]
            label_onehot=np.zeros(n_classes,dtype=np.float32)
            label_name_index=key_classes.index(filename.split('/')[0])
            label_onehot[label_name_index]=1
            images_batch.append(image_ret)
            labels_batch.append(label_onehot)    
    return images_batch,labels_batch   

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
     
def get_AllTestingImage(imageDataDict,image_names,n_classes,key_classes):
    images_batch=[]
    labels_batch=[]
    image_names_len=len(image_names)
    for filename in image_names:
        if re.search('[jpg|jpeg|JPG|JPEG|png|PNG]',filename):
            image_ret=imageDataDict[filename]
            label_onehot=np.zeros(n_classes,dtype=np.float32)
            label_name_index=key_classes.index(filename.split('/')[0])
            label_onehot[label_name_index]=1
            images_batch.append(image_ret)
            labels_batch.append(label_onehot) 
    return images_batch,labels_batch  

def get_RandomTestingImage(imageDataDict,image_names,n_classes,key_classes,random_batch):
    images_batch=[]
    labels_batch=[]
    image_names_len=len(image_names)
    for i in range(random_batch):
        filename=image_names[np.random.randint(image_names_len)]
        if re.search('[jpg|jpeg|JPG|JPEG|png|PNG]',filename):
            image_ret=imageDataDict[filename]
            label_onehot=np.zeros(n_classes,dtype=np.float32)
            label_name_index=key_classes.index(filename.split('/')[0])
            label_onehot[label_name_index]=1
            images_batch.append(image_ret)
            labels_batch.append(label_onehot)    
    return images_batch,labels_batch 

def get_RandomTestingImage_withoutDict(image_dir,image_names,n_classes,key_classes,random_batch):
    images_batch=[]
    labels_batch=[]
    filenames=[]
    image_names_len=len(image_names)
    for i in range(random_batch):
        filename=image_names[np.random.randint(image_names_len)]
        filenames.append(filename)
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
    return images_batch,labels_batch,filenames

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

def get_imageDataDict(image_dir,image_names):
    imageDataDict={}
    for filename in image_names:
        image_path=os.path.join(image_dir,filename)
        if not os.path.exists(image_path):
            continue
        if re.search('[jpg|jpeg|JPG|JPEG|png|PNG]',filename): 
            image_tmp=cv2.imread(image_path)
            image_tmp=image_tmp*(1.0/255)-0.5
            image_ret=image_tmp.astype(np.float32)
            imageDataDict[filename]=image_ret
    return imageDataDict


def train():
    # trainingLabelTxt='training_labels.txt'
    # validationLabelTxt='validation_labels.txt'
    # testingLabelTxt='testing_labels.txt'
    image_lists=create_image_list(TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
    WriteImageLabelTxt(image_lists,INPUT_DATA)
    n_classes=len(image_lists)
    key_classes=image_lists.keys()

    training_image_names=get_AllLabel(INPUT_DATA,trainingLabelTxt)
    testing_image_names=get_AllLabel(INPUT_DATA,testingLabelTxt)
    validation_image_names=get_AllLabel(INPUT_DATA,validationLabelTxt)

    # training_imageDataDict=get_imageDataDict(INPUT_DATA,training_image_names)
    # testing_imageDataDict=get_imageDataDict(INPUT_DATA,testing_image_names)
    # validation_imageDataDict=get_imageDataDict(INPUT_DATA,validation_image_names)
    with open(os.path.join(INPUT_DATA,category_txt),'w') as f:
        for classname in key_classes:
            f.write(classname+'\n')

    print "The number of class is:%d"%(n_classes)
    print 'The number of Training image:%d'%(len(training_image_names))
    print 'The number of testing image:%d'%(len(testing_image_names))
    print 'The number of validation image:%d'%(len(validation_image_names))
    images_input=tf.placeholder(tf.float32,[None,227,227,3],name='images_input')
    labels_input=tf.placeholder(tf.float32,[None,n_classes],name='labels_input')
    #learning_rate=tf.placeholder(tf.float32,[1,],name='learning_rate')
    net=Alexnet.network(n_classes)
    logits=net.inference(images_input)
    loss=net.softmax_loss(labels_input,logits)
    evalution=net.evalution(logits,labels_input)
    Maxpro,Maxpro_index=net.MaxproAndindex(logits)
    saver=tf.train.Saver()
    
    f_train=open(trainingLog,'w')
    f_validation=open(validationLog,'w')
    f_testing=open(testingLog,'w')
    f_train.close()
    f_validation.close()
    f_testing.close()
    with tf.Session() as sess:
        t0=time.time()
        init=tf.global_variables_initializer()
        sess.run(init)
        print '===================== Start Training ====================>'
        for i in range(STEPS):
            global LEARNING_RATE
            time_now=datetime.datetime.now()
            #images_batch,labels_batch=get_nextbatchImage(training_imageDataDict,training_image_names,n_classes,key_classes,i,BATCH)
            images_batch,labels_batch=get_nextbatchImage_withoutDict(INPUT_DATA,training_image_names,n_classes,key_classes,i,BATCH)
            LEARNING_RATE=LEARNING_RATE/(10**(i/2000))
            if LEARNING_RATE<0.0001:
                LEARNING_RATE=0.0001
            #print LEARNING_RATE
            opti=net.optimer(loss,LEARNING_RATE)
            #images_batch,labels_batch=get_RandomImage_BatchData(image_lists,INPUT_DATA,n_classes,BATCH,'training')
            with tf.device("/gpu:0"):
                trainloss,trainevalution,_=sess.run([loss,evalution,opti],feed_dict={images_input:images_batch,labels_input:labels_batch})
            with open(trainingLog,'a') as f_train:
                f_train.write(str(time_now).split('.')[0]+'    step:'+str(i)+'   accuracy:'+str(trainevalution)+'    loss:'+str(trainloss)+'\n')
            if i%100==0 or i+1==STEPS:
                #Validtionimages_batch,Validtionlabels_batch=get_nextbatchImage(validation_imageDataDict,validation_image_names,n_classes,key_classes,i/100-1,BATCH)
                Validtionimages_batch,Validtionlabels_batch=get_nextbatchImage_withoutDict(INPUT_DATA,validation_image_names,n_classes,key_classes,i/100-1,BATCH)
                #Validtionimages_batch,Validtionlabels_batch=get_RandomImage_BatchData(image_lists,INPUT_DATA,n_classes,BATCH,'validation')
                with tf.device("/gpu:0"):
                    eva_loss_np,eva_evalution_np=sess.run([loss,evalution],feed_dict={images_input:Validtionimages_batch,labels_input:Validtionlabels_batch})
                with open(validationLog,'a') as f_validation:
                    f_validation.write(str(time_now).split('.')[0]+'    step:'+str(i)+'   accuracy:'+str(eva_evalution_np)+'    loss:'+str(eva_loss_np)+'\n')
                print (str(time_now).split('.')[0]+'  Step:%d  Validation on random sample:%d =====>Accuracy:%0.2f%%  Loss:%0.2f'%(i,BATCH,eva_evalution_np*100,eva_loss_np))
        
            if i!=0 and i%2000==0 or i+1==STEPS:
                print '===================== Start Saving Model ====================>'
                model_t0=time.time()
                if not os.path.exists(MODELSAVE_DIR):
                    os.mkdir(MODELSAVE_DIR)
                print 'the model is saved in :' + MODELSAVE_DIR
                print 'the model name is: '+'AlexnetModel_'+str(i)
                saver.save(sess,MODELSAVE_DIR+'/AlexnetModel_'+str(i))
                print '===================== Finish Saving Model ====================>'
                print 'Saving Time is : %0.2fs\n'%(time.time()-model_t0) 

        print '===================== Finish Training ====================>'
        print 'Training Time is : %0.2fs\n'%(time.time()-t0)
            
        print '===================== Start Testing ====================>'
        random_batch=100
        Testimages_batch,Testlabels_batch,filenames=get_RandomTestingImage_withoutDict(INPUT_DATA,testing_image_names,n_classes,key_classes,random_batch)
        #Testimages_batch,Testlabels_batch=get_RandomTestingImage(testing_imageDataDict,testing_image_names,n_classes,key_classes,random_batch)
        #Testimages_batch,Testlabels_batch=get_AllTestingImage(testing_imageDataDict,testing_image_names,n_classes,key_classes)
        #Testimages_batch,Testlabels_batch=get_AllTestingImage(INPUT_DATA,testingLabelTxt,n_classes,key_classes)
        #Testimages_batch,Testlabels_batch=get_AllImageTest(image_lists,INPUT_DATA,n_classes)
        with tf.device("/gpu:0"):
            test_loss_np,test_evalution_np,testMaxpro,testMaxpro_index=sess.run([loss,evalution,Maxpro,Maxpro_index],feed_dict={images_input:Testimages_batch,labels_input:Testlabels_batch})
        with open(testingLog,'a') as f_testing:
            f_testing.write(str(time_now).split('.')[0]+'   accuracy:'+str(test_evalution_np)+'    loss:'+str(test_loss_np)+'\n')
        test_t0=time.time()
        for i in range(len(testMaxpro_index)):
            TrueLabel=filenames[i].split('/')[0]
            PredictLabel=key_classes[testMaxpro_index[i]]
            print 'image%d: Imagename:%s TrueLabel:%s PredictLabel:%s PredictPro:%0.2f%% Result:%s \n'%(i+1,filenames[i],TrueLabel,PredictLabel,testMaxpro[i]*100,str(PredictLabel==TrueLabel))
        #print 'The number of testImage is :%d'%(len(Testlabels_batch))
        #print 'Testing Accuracy:%0.2f%%  Loss:%0.2f:'%(test_evalution_np*100,test_loss_np)
        print ('Testing on random sample:%d =====>Accuracy:%0.2f%%  Loss:%0.2f'%(random_batch,test_evalution_np*100,test_loss_np))
        print '===================== Finish Testing ====================>'
        print 'Testing Time is : %0.2fs\n'%(time.time()-test_t0)
            

            

if __name__ == '__main__':
    train()


