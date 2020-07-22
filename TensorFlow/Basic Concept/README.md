#!/usr/bin/env python
# coding: utf-8

# # Introdution to TensorFlow for AI
# #https://www.coursera.org/learn/introduction-tensorflow/lecture/kr51q/the-hello-world-of-neural-networks

# # Week 1 - Introduction
# 

# In[48]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers


# ## 1. Neural Network Easy Example

# In[4]:


model=keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])


# In[5]:


model.compile(optimizer='sgd',loss='mean_squared_error')


# In[11]:


xs=np.array([-1.0,0.0,1.0,2.0,3.0,4.0],dtype=float)
ys=np.array([-3.0,-1.0,1.0,3.0,5.0,7.0],dtype=float)


# In[19]:


model.fit(xs,ys,epochs=500)


# ### 会接近19但不会是19的原因

# 首先是您使用很少的数据对其进行了训练。只有六点。这六个点是线性的，但是不能保证每个X的关系都是Y等于2X减去1。对于X等于10，Y等于19的可能性很高，但是神经网络不是正数。因此，它将得出Y的实际值。这是第二个主要原因。当使用神经网络时，由于他们试图找出所有问题的答案，所以他们要处理概率。

# You might have thought 19, right? But it ended up being a little under. Why do you think that is? 
# 
# Remember that neural networks deal with probabilities, so given the data that we fed the NN with, it calculated that there is a very high probability that the relationship between X and Y is Y=2X-1, but with only 6 data points we can't know for sure. As a result, the result for 10 is very close to 19, but not necessarily 19. 
# 
# As you work with neural networks, you'll see this pattern recurring. You will almost always deal with probabilities, not certainties, and will do a little bit of coding to figure out what the result is based on the probabilities, particularly when it comes to classification.
# 

# In[20]:


print(model.predict([10.0]))


# ## 2. Exerice - Predict House Price

# In[23]:


model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0,11.0, 12.0, 13.0], dtype=float)
ys = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 450.0, 500.0, 550.0,600.0, 650.0,700.0], dtype=float)
model.fit(xs,ys,epochs=500)


# In[24]:


print(model.predict([7.0]))


# # Week 2 - Computer Vision

# In[31]:


# Fashion MNIST
fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()


# In[32]:


model= keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                        keras.layers.Dense(128,activation=tf.nn.relu),
                        keras.layers.Dense(10,activation=tf.nn.softmax)])


# In[35]:


mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


# In[38]:


np.set_printoptions(linewidth=200)
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])


# In[39]:


# normalizing
training_images  = training_images / 255.0
test_images = test_images / 255.0


# In[40]:


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


# **Sequential**: That defines a SEQUENCE of layers in the neural network
# 
# **Flatten**: Remember earlier where our images were a square, when you printed them out? Flatten just takes that square and turns it into a 1 dimensional set.
# 
# **Dense**: Adds a layer of neurons
# 
# Each layer of neurons need an **activation function** to tell them what to do. There's lots of options, but just use these for now. 
# 
# **Relu** effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.
# 
# **Softmax** takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!
# 

# In[52]:


# model.compile(optimizer = tf.optimizers.Adam(),
#               loss = 'sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(training_images, training_labels, epochs=5)

model.compile(optimizer = 'Adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)


# In[53]:


model.evaluate(test_images, test_labels)


# ### Call back

# 回调函数是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。通过传递回调函数列表到模型的.fit()中，即可在给定的训练阶段调用该函数集中的函数。
# 
# 【Tips】虽然我们称之为回调“函数”，但事实上Keras的回调函数是一个类，回调函数只是习惯性称呼

# # Week 3 - Convolutional Neural Networks
# 

# In[4]:


import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
# 加入convolutional + max pooling
model = tf.keras.models.Sequential([
    # 第一层共64个3*3的filter，activation是relu，输入层是28*28*1，输出的28-3+1=26*26*64
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    # maxpooling层，每2*2个里面挑选max的出来，输出的是 13*13*64
    tf.keras.layers.MaxPooling2D(2,2),
    # 再加一层，13-3+1=11*11*64
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    # 输出 5*5*64=1600
    tf.keras.layers.MaxPooling2D(2,2),
    # 输出1600
    tf.keras.layers.Flatten(),
    # 输出128
    tf.keras.layers.Dense(128,activation='relu'),
    # 输出10
    tf.keras.layers.Dense(10,activation='softmax'),  
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 用来了解每一步的情况
model.summary()
model.fit(training_images, training_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)


# ## Visualizing the Convolutions and Pooling
# 
# This code will show us the convolutions graphically. The print (test_labels[;100]) shows us the first 100 labels in the test set, and you can see that the ones at index 0, index 23 and index 28 are all the same value (9). They're all shoes. Let's take a look at the result of running the convolution on each, and you'll begin to see common features between them emerge. Now, when the DNN is training on that data, it's working with a lot less, and it's perhaps finding a commonality between shoes based on this convolution/pooling combination.

# In[5]:


print(test_labels[:100])


# In[7]:


import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 1
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
    f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0,x].grid(False)
    f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1,x].grid(False)
    f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2,x].grid(False)


# In[ ]:




