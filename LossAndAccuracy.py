import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np 
import re
import os
import shutil

trainingLog='LogFile/training_log1.log'
testingLog='LogFile/testing_log1.log'
validationLog='LogFile/validation_log1.log'

txtname=[trainingLog,testingLog,validationLog]
for i in range(3):
    if os.path.exists(txtname[i]):
        shutil.copy(txtname[i],txtname[i].split('.')[0]+'.txt')

trainingTxt=trainingLog.split('.')[0]+'.txt'
trainingAccuracy=[]
trainingLoss=[]
with open(trainingTxt,'r') as f:
    for line in f:
        tmp=line.split('   ')
        #print tmp
        trainingAccuracy.append(float(tmp[2].split(':')[1]))
        trainingLoss.append(float(tmp[3].split(':')[1]))

#coding=utf-8
fig = plt.figure()
ax1 = fig.add_subplot(111)
steps=range(len(trainingAccuracy))

ax1.plot(steps,trainingLoss,'b',label='loss')
ax1.set_xlabel('step')
ax1.set_ylabel('loss')
ax1.legend(loc=2)
ax2 = ax1.twinx() 
ax2.plot(steps,trainingAccuracy,'r',label='accuracy')
ax2.set_xlabel('step')
ax2.set_ylabel('accuracy')
ax2.legend(loc=1)

plt.title('Loss And Accuracy')
plt.grid()
plt.show()