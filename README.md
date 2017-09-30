## 1.项目简介

这是利用深度学习做的一个比较简单的人脸识别，训练的框架是AlexNet。本项目对376个类进行分类识别，每个类代表一个人的脸，每个类里有关于这个人的若干张人脸， 把每个类里80%的照片用来训练，10%的照片用来验证，10%的照片用来测试，然后用设置好的神经网络对训练集进行训练并验证，最后用训练保存好的模型对测试集进行测 试，求出这个模型对人脸识别的准确率。由于后期没做进一步的优化，识别的准确率在70%左右。
## 2.ALexNet简介

AlexNet是在2012年被发表的一个金典之作，并在当年取得了ImageNet最好成绩，也是在那年之后，更多的更深的神经网路被提出， 比如优秀的VGG,GoogleLeNet.其官方提供的数据模型，准确率达到57.1%,top 1-5 达到80.2%. 这项对于传统的机器学习分类算法而言，已经相当的出色。<br>
![iamge](https://github.com/Lihit/FaceRecognition/blob/master/ResultImage/AlexNet.png)
## 3.运行环境
* ubuntu 14.04
* 需要有GPU和安转CUDA。
* 编程语言是Python，推荐安装Anaconda,需要安装的python库：
    * opencv
    * tensorflow
    * numpy

## 4.使用方式

* 将项目clone到你的本地：`git clone https://github.com/Lihit/FaceRecognition.git`
* 到下载后的文件夹`：cd FaceRecognition`。
* 文件`AlexNet.py`是AlexNet的具体实现，讲它写成了一个类。
* 打开终端，运行`python AlexnetTrain.py`，即可进行训练，但是由于训练数据太大，无法上传，需要自己准备数据或是联系我。
* 如果你是自己准备数据，可以使用`python ExpandImageSample.py`来扩充你的数据集，具体的做法以及要修改的一些地方（如文件路径），请 
看`ExpandImageSample.py` 的代码。
* 如果能够正常训练，那么文件夹`MyModel`将会保存你训练的模型（如果你想下载我训练模型，请联系我），而文件夹`LogFIle`将会保存你的训练日志。
* `AlexnetTest.ipynb`用来测试准确率，`LossAndAccuracy.ipynb`用来绘制损失和准确率函数。

## 5.运行结果

* 训练时的训练误差和验证误差：<br>
![iamge](https://github.com/Lihit/FaceRecognition/blob/master/ResultImage/traingingErro.png)
* 单人的测试结果：<br>
![iamge](https://github.com/Lihit/FaceRecognition/blob/master/ResultImage/testResult.png)
* 随机抽取20人的测试结果：<br>
![iamge](https://github.com/Lihit/FaceRecognition/blob/master/ResultImage/testResult(20pictures).png)
* loss和Accuracy曲线的绘制：<br>
![iamge](https://github.com/Lihit/FaceRecognition/blob/master/ResultImage/lossAndAccuracy.png)
