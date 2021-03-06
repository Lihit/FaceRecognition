import tensorflow as tf
import  cv2 
import  os 

class network(object): 

    def __init__(self,n_classes): 
        with tf.variable_scope("weights"):
           self.weights={ 

                #39*39*3->36*36*20->18*18*20 

                'conv1':tf.get_variable('conv1',[11,11,3,96],initializer=tf.contrib.layers.xavier_initializer_conv2d()), 

                #18*18*20->16*16*40->8*8*40 

                'conv2':tf.get_variable('conv2',[5,5,96,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()), 

                #8*8*40->6*6*60->3*3*60 

                'conv3':tf.get_variable('conv3',[3,3,256,384],initializer=tf.contrib.layers.xavier_initializer_conv2d()), 

                #3*3*60->120 

                'conv4':tf.get_variable('conv4',[3,3,384,384],initializer=tf.contrib.layers.xavier_initializer_conv2d()), 

 
                'conv5':tf.get_variable('conv5',[3,3,384,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()), 

 
                'fc1':tf.get_variable('fc1',[6*6*256,4096],initializer=tf.contrib.layers.xavier_initializer()), 

                'fc2':tf.get_variable('fc2',[4096,4096],initializer=tf.contrib.layers.xavier_initializer()), 


                #120->6 

                'fc3':tf.get_variable('fc3',[4096,n_classes],initializer=tf.contrib.layers.xavier_initializer()), 

                } 

        with tf.variable_scope("biases"): 

            self.biases={ 

                'conv1':tf.get_variable('conv1',[96,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)), 

                'conv2':tf.get_variable('conv2',[256,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)), 

                'conv3':tf.get_variable('conv3',[384,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)), 

                'conv4':tf.get_variable('conv4',[384,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)), 

                'conv5':tf.get_variable('conv5',[256,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)), 

               

                'fc1':tf.get_variable('fc1',[4096,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)), 

                'fc2':tf.get_variable('fc2',[4096,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'fc3':tf.get_variable('fc3',[n_classes,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)) 

            }

    def inference(self,images): 
        # images = tf.reshape(images, shape=[-1, 39,39, 3])
        images = tf.reshape(images, shape=[-1, 227,227, 3])# [batch, in_height, in_width, in_channels] 
        #images=tf.cast(images,tf.float32)
        conv1=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 4, 4, 1], padding='VALID'),self.biases['conv1']) 
        relu1= tf.nn.relu(conv1) 
        pool1=tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID') 

        conv2=tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='SAME'), self.biases['conv2']) 
        relu2= tf.nn.relu(conv2) 
        pool2=tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID') 

        conv3=tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='SAME'), self.biases['conv3']) 
        relu3= tf.nn.relu(conv3) 

        # pool3=tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') 

        conv4=tf.nn.bias_add(tf.nn.conv2d(relu3, self.weights['conv4'], strides=[1, 1, 1, 1], padding='SAME'), self.biases['conv4']) 
        relu4= tf.nn.relu(conv4)
        conv5=tf.nn.bias_add(tf.nn.conv2d(relu4, self.weights['conv5'], strides=[1, 1, 1, 1], padding='SAME'), self.biases['conv5']) 
        relu5= tf.nn.relu(conv5)
        pool5=tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        flatten = tf.reshape(pool5, [-1, self.weights['fc1'].get_shape().as_list()[0]]) 
        drop1=tf.nn.dropout(flatten,0.5) 
        fc1=tf.matmul(drop1, self.weights['fc1'])+self.biases['fc1'] 
        fc_relu1=tf.nn.relu(fc1) 

        fc2=tf.matmul(fc_relu1, self.weights['fc2'])+self.biases['fc2']
        fc_relu2=tf.nn.relu(fc2) 
        #fc3=tf.matmul(fc_relu2, self.weights['fc3'])+self.biases['fc3']
        fc3=tf.add(tf.matmul(fc_relu2, self.weights['fc3']),self.biases['fc3'],name='logits')
        return  fc3 

    def inference_test(self,images): 

        images = tf.reshape(images, shape=[-1, 39,39, 3])# [batch, in_height, in_width, in_channels] 

        #images=(tf.cast(images,tf.float32)/255.-0.5)*2

        conv1=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='VALID'), self.biases['conv1']) 
        relu1= tf.nn.relu(conv1) 
        pool1=tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') 


        conv2=tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='VALID'), self.biases['conv2']) 
        relu2= tf.nn.relu(conv2) 
        pool2=tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') 
 
        conv3=tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID'),  self.biases['conv3']) 
        relu3= tf.nn.relu(conv3) 
        pool3=tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') 

        flatten = tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]]) 
        fc1=tf.matmul(flatten, self.weights['fc1'])+self.biases['fc1'] 
        fc_relu1=tf.nn.relu(fc1) 
        #fc2=tf.matmul(fc_relu1, self.weights['fc2'])+self.biases['fc2']
        fc2=tf.add(tf.matmul(fc_relu1, self.weights['fc2']),self.biases['fc2'],name='test_logits')
        return  fc2 

    def softmax_loss(self,labels_input,logits): 
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=labels_input,logits=logits)
        cross_entropy_mean=tf.reduce_mean(cross_entropy,name='cross_entropy_mean')
        #predicts=tf.nn.softmax(predicts) 
        #labels=tf.one_hot(labels,self.weights['fc3'].get_shape().as_list()[1]) 
        #loss = tf.nn.softmax_cross_entropy_with_logits(predicts, labels)
        #loss =-tf.reduce_mean(labels * tf.log(predicts))# tf.nn.softmax_cross_entropy_with_logits(predicts, labels) 
        self.cost= cross_entropy_mean 
        return self.cost  

    def optimer(self,cross_entropy_mean,lr=0.01): 
        train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy_mean,name='train_optimizer') 
        return train_optimizer 

    def evalution(self,logits,labels_input):
        predicts=tf.nn.softmax(logits)
        correct_prediction=tf.equal(tf.argmax(predicts,1),tf.argmax(labels_input,1))
        evalution=tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='evalution')
        return evalution
    
    def MaxproAndindex(self,logits):
        predicts=tf.nn.softmax(logits)
        Maxpro=tf.reduce_max(predicts,1,name='Maxpro')
        Maxpro_index=tf.argmax(predicts,1,name='Maxpro_index')
        return Maxpro,Maxpro_index
        
