import cv2
import numpy as np
import os
import re
import copy
import shutil
import time

img_width=227
img_heigth=227

def Image_RotateLeftAndRight(image_origin):
    if image_origin is None:
        print 'Error:func Image_RotateLeftAndRight:please input a valid image!'
        return None
    image_rotation=np.zeros(image_origin.shape,dtype=np.uint8)
    image_rotation[:,::-1,:]=image_origin[:,:,:]
    return cv2.resize(image_rotation,(img_heigth,img_width))


def Image_Crop(image_origin):
    CropCoef=0.8
    if image_origin is None:
        print 'Error:func Image_Crop:please input a valid image!'  
        return None
    img_shape=image_origin.shape
    #print img_shape
    img_width=img_shape[1]
    img_heigth=img_shape[0]
    width_start=int(img_width*(1-CropCoef)/2)
    width_end=int(img_width*(1-(1-CropCoef)/2))
    heigth_start=int(img_heigth*(1-CropCoef)/2)
    heigth_end=int(img_heigth*(1-(1-CropCoef)/2))
    image_crop=np.zeros((heigth_end-heigth_start,width_end-width_start,3),dtype=np.uint8)
    image_crop=image_origin[heigth_start:heigth_end,width_start:width_end,:]
    #image_crop=image_crop_tmp.reshape(img_shape)
    return cv2.resize(image_crop,(img_heigth,img_width))


def Image_Darker(image_origin):
    if image_origin is None:
        print 'Error:func Image_Crop:please input a valid image!'  
        return None
    image_darker=np.zeros(image_origin.shape,dtype=np.uint8)
    image_darker=copy.copy(image_origin)
    image_darker[:,:,2]=image_darker[:,:,2]*0.8
    return cv2.resize(image_darker,(img_heigth,img_width))

def Image_Lighter(image_origin):
    if image_origin is None:
        print 'Error:func Image_Crop:please input a valid image!'  
        return None
    image_ligther=np.zeros(image_origin.shape,dtype=np.uint8)
    image_ligther=copy.copy(image_origin)
    image_ligther[:,:,2]=image_ligther[:,:,2]*1.1
    return cv2.resize(image_ligther,(img_heigth,img_width))

def ImageRandomWrite(image_origin,rotation_flag,crop_flag,darker_flag,lighter_flag):
    image_ret=copy.copy(image_origin)
    if rotation_flag==1:
        image_ret=Image_RotateLeftAndRight(image_ret)
    if crop_flag==1:
        image_ret=Image_Crop(image_ret)
    if darker_flag==1:
        image_ret=Image_Darker(image_ret)
    if lighter_flag==1:
        image_ret=Image_Lighter(image_ret)
    return cv2.resize(image_ret,(img_heigth,img_width))

def ExpandImageSample(sub_dir,sub_dir_create):
    filenum=len(os.listdir(sub_dir))
    image_expand_num=100//filenum
    for imagename in os.listdir(sub_dir):
        if re.search('[jpg|JPG|PNG|png|jpeg|JPEG]',imagename):
            image_origin=cv2.imread(os.path.join(sub_dir,imagename))
            print os.path.join(sub_dir,imagename)
            cv2.imwrite(os.path.join(sub_dir_create,imagename),cv2.resize(image_origin,(img_heigth,img_width)))
            for i in range(image_expand_num):
                image_ret=ImageRandomWrite(image_origin,(i+1)%2,((i+1)/2)%2,((i+1)/4)%2,((i+1)/8)%2)
                cv2.imwrite(os.path.join(sub_dir_create,imagename.split('.')[0]+str(i+1)+'.'+imagename.split('.')[1]),image_ret)


def ImageReWrite(image_path,image_create_path):
    if not os.path.isdir(image_path):
        print 'error:Please input correct image_path!'
        return 
    for filename in os.listdir(image_path):
        sub_dir=os.path.join(image_path,filename)
        sub_dir_create=os.path.join(image_create_path,filename)
        if not os.path.isdir(sub_dir):
            continue
        if os.path.exists(sub_dir_create):
            shutil.rmtree(sub_dir_create)
        os.mkdir(sub_dir_create)
        if len(os.listdir(sub_dir))<7:
            continue
        if len(os.listdir(sub_dir))>100:
            for imagename in os.listdir(sub_dir):
                if re.search('[jpg|JPG|PNG|png|jpeg|JPEG]',imagename):
                    print os.path.join(sub_dir,imagename)
                    cv2.imwrite(os.path.join(sub_dir_create,imagename),cv2.resize(cv2.imread(os.path.join(sub_dir,imagename)),(img_heigth,img_width)))
        else:
            ExpandImageSample(sub_dir,sub_dir_create)


def main_test(image_path):
    img_origin=cv2.imread(image_path)
    cv2.imshow('origin image',img_origin)
    #image_rotation=Image_RotateLeftAndRight(img_origin)
    #image_crop=Image_Crop(img_origin)
    image_darker=Image_Darker(img_origin)
    #image_ligther=Image_Lighter(img_origin)
    cv2.imshow('image_ligther',image_darker)
    cv2.waitKey(0)

if __name__ == '__main__':
    t0=time.time()
    print '=======================Start Processing=======================>'
    image_path='./data/data1'
    image_create_path='./data/mydata1'
    if not os.path.exists(image_create_path):
        os.mkdir(image_create_path)
    ImageReWrite(image_path,image_create_path)
    print '=======================Finish Processing======================>'
    print 'The total time of processing is:%0.2fs'%(time.time()-t0)
    #image_path='/home/wenshao/TensorFlow/Tf_MigrationLearnning/1.jpg'
    #main_test(image_path)




