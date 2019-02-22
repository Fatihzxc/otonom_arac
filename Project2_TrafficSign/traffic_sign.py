import pickle

training_file = 'data/train.p'
validation_file = 'data/validate.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f,encoding='latin1')
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f, encoding='latin1')
with open(testing_file, mode='rb') as f:
    test = pickle.load(f, encoding='latin1')

    
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

import numpy as np

X_train = np.array(X_train)
X_valid = np.array(X_valid)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_valid = np.array(y_valid)
y_test = np.array(y_test)

X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_valid = np.pad(X_valid, ((0,0),(2,2),(2,2),(0,0)),'constant')
X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

n_train = X_train.shape[0]
n_test = X_test.shape[0]
image_shape = X_train.shape[1:]
n_classes = len(np.unique(y_train))

print("Number of training examples ", n_train)
print("Number of testing examples ", n_test)


### Preprocess images: normalize images from [0,255] to [0,1], and gray scale it
### Rationale: although colors in the traffic sign are important in real world for people to recoganize
###            different signs, traffic signs are also different in their shapes and contents. We can
###            ignore colors in this problem because signs in our training set are differentiable from
###            their contents and shapes.
import numpy as np
import cv2

def reshape_raw_images(imgs):
    """Given 4D images (number, heigh, weight, channel), this
    function grayscales and returns (number, height, weight, 1) images"""
    def gray(src):
        if src.dtype == np.uint8:
            src = np.array(src/255.0, dtype=np.float32)
        dst = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        return dst.reshape(32,32,1)
    norms = [gray(img) for img in imgs]
    return np.array(norms)
    
X_train = reshape_raw_images(X_train)
labels_train   = y_train

X_valid = reshape_raw_images(X_valid)
labels_valid   = y_valid

X_test  = reshape_raw_images(X_test)
labels_test    = y_test

### Architecture:
###   I adapted LeNet architecture: Two convolutional layers followed by one flatten layer and three
###   fully connected linear layers.
###
###   convolution 1: 32x32x1  -> 28x28x12 -> relu -> 14x14x12 (pooling)
###   convolution 2: 14x14x12 -> 10x10x25 -> relu -> 5x5x25   (pooling)
###         flatten: 5x5x25   -> 625
###        drop out: 625      -> 625
###          linear: 625      -> 300
###          linear: 300      -> 150
###          linear: 150      -> 43

### Experiment shows that
### 1. drop out has positive impact on the accuracy
### 2. longer depth (to some extent) on convolution filter has better accuracy

import tensorflow as tf

mu, sigma = 0, 0.1

def conv(input, in_len, in_depth, out_len, out_depth):
    """ Define a convolutional network layer
    @param input: input data or model
    @param in_len: 0D, the input height and width (assume they're the same)
    @param in_depth: 0D, the input depth (e.g. 3 for RGB images)
    @param out_len: 0D, desired output height and width
    @param out_depth: 0D, desired output depth
    """
    filter_len = in_len - out_len + 1
    # we're not going to use stride to reduce the dimention,
    # we're going to use pooling instead.
    strides = [1,1,1,1]

    W = tf.Variable(tf.truncated_normal(shape=(filter_len, filter_len, in_depth, out_depth), \
                                        mean = mu, stddev = sigma))
    b = tf.Variable(tf.zeros(out_depth))
    model = tf.nn.conv2d(input, W, strides=strides, padding='VALID') + b
    return model

def linear(input, in_size, out_size):
    W = tf.Variable(tf.truncated_normal(shape=(in_size, out_size), \
                                        mean=mu, stddev = sigma))
    b = tf.Variable(tf.zeros(out_size))
    model = tf.matmul(input, W) + b
    return model
    
def classifier(input, keep_prob):
    ## Layer1
    # convolution layer: 32x32x1 -> 28x28x12
    conv1 = conv(input, 32, 1, 28, 12)
    conv1 = tf.nn.relu(conv1)
    # pooling: 28x28x12 -> 14x14x12
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    ## Layer2
    # convolution layer 14x14x12 -> 10x10x25
    conv2 = conv(conv1, 14, 12, 10, 25)
    conv2 = tf.nn.relu(conv2)
    # pooling: 10x10x25 -> 5x5x25
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    ## Layer3
    # -> 625
    flat = tf.contrib.layers.flatten(conv2)
    dropped = tf.nn.dropout(flat, keep_prob)
    layer3 = linear(dropped, 625, 300)
    layer3 = tf.nn.relu(layer3)
    
    ## Layer4
    layer4 = linear(layer3, 300, 100)
    layer4 = tf.nn.relu(layer4)
    
    ## Layer 5
    layer5 = linear(layer4, 100, n_classes)
    return layer5

### Define the model and its input

from sklearn.utils import shuffle

feature_shape = X_train.shape[1:]
x = tf.placeholder(tf.float32, shape=(None,)+feature_shape)
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
# drop out layer parameter. This parameter should be 1.0 when evaluate and test
# the model, less than 1.0 when training
keep_prob = tf.placeholder(tf.float32)

logits = classifier(x, keep_prob)

# A function for evaluating the accuracy of a model
prediction_step = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuary_step = tf.reduce_mean(tf.cast(prediction_step, tf.float32))
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuary_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


import time

saver = tf.train.Saver()
model_file = './model/model'

epoch=10
batch_size = 64
rate = 0.001

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate = rate).minimize(loss)

X_train, y_train = shuffle(X_train, y_train)
def train(X_data, y_data): 
  with tf.Session() as ses:
    ses.run(tf.global_variables_initializer())
    print('Starting...')
    for i in range(epoch):
      begin_time = time.time()

      for offset in range(0, len(X_train), batch_size):
        end= offset+batch_size

        features, labels = X_train[offset:end], y_train[offset:end]
        ses.run(train_step, feed_dict={x:features, y:labels, keep_prob:0.8})
      validation = evaluate(X_valid, y_valid)
      print("[{3:.1f}s] epoch {0}/{1}: validation = {2:.3f}".format(i+1, epoch, validation, time.time()-begin_time))

      saver.save(ses, model_file)
      print('Model saved')

train(X_train, y_train)


with tf.Session() as ses:
  saver.restore(ses, model_file)
  acc = evaluate(X_test, y_test)
  print('Accuracy is {}'.format(acc))


"""
from PIL import Image
import os

def read_file_to_32x32_array(file):
    x = Image.open(file).convert("RGB")
    x = x.resize((32,32))
    return np.array(x)


def predict(features, human_readable=False):
    with tf.Session() as sess:
        saver.restore(sess, model_file)
        results = sess.run(tf.argmax(logits, 1), {x : features, keep_prob:1.0})
        if human_readable:
            results = [signname[n] for n in results]
        return results


while True:
  yol = input('Resim yolu giriniz... ')
	my_images = np.array([read_file_to_32x32_array(yol)])
	my_features = reshape_raw_images(my_images)
	plt.plot(my_images)

	predict(my_feature, True)  
"""
