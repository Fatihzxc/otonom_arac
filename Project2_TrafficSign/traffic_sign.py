import pickle

training_file   =   'data/train.p'
validation_file =   'data/validate.p'
testing_file    =   'data/test.p'

with open(training_file, mode='rb') as f:
  train = pickle.load(f, encoding='latin1')

with open(validation_file, mode='rb') as f:
  valid = pickle.load(f, encoding='latin1')

with open(testing_file, mode='rb') as f:
  test = pickle.load(f, encoding='latin1')

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test   = test['features'], test['labels']

import numpy as np
X_train = np.array(X_train)
X_valid = np.array(X_valid)
X_test = np.array(X_test)


n_train = X_train.shape[0] 
n_validation= X_valid.shape[0]
n_test = X_test.shape[0]
image_shape = X_train.shape[1:] 
n_classes = len(np.unique(y_train)) 
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import matplotlib.pyplot as plt
#plt.imshow(X_train[0], cmap='gray')
#plt.show()

# 28x28 to 32x32
X_train= np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test= np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)),'constant')
X_valid= np.pad(X_valid, ((0,0),(2,2),(2,2),(0,0)),'constant')

#TODO GRAYSCALE DONUSTUR
# GrayScale Donusum islemleri bu sekildedir.
X_train = np.sum(X_train/3, axis=3, keepdims=True)
X_test = np.sum(X_test/3, axis=3, keepdims=True)
X_valid = np.sum(X_valid/3, axis=3, keepdims=True)
#print('Converted to grayscale, the shape is : ', X_train.shape)


#TODO VERI SETINI NORMALIZE ET
# Normalize islemi
X_train_normalized =  (X_train - 128)  / 128
#x_valid =  (X_valid - 128)  / 128
X_test_normalized =   (X_test - 128)   / 128

#plt.imshow(X_train_normalized[0].squeeze(), cmap='gray')
#plt.show()

#TODO - RANDOM_TRANSLATE

#TODO - RANDOM_SCALING

#TODO - RANDOM_WARP

#TODO - RANDOM_BRIGHTNESS






#TODO VERI SETINI KARISTIR
# Veri Setini Karistirma Islemi
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)


#TODO LENET MODELINI KUR
import tensorflow as tf
EPOCHS = 10
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten

def LeNet(x):
  mu = 0
  sigma = 0.1
  
  # NOT: x_shape [2,3] diyelim ve 1 kanal
  # NOT: valid_pad = maxpool 2x2 kernel, 2-stride, VALID padding->shape[1,1]
  # NOT: same_pad = maxpool 2x2 kernel, 2-stride, SAME padding->shape[2,4]

  # Layer 1 ---------- 3->RGB 1->GRAY
  conv1_W = tf.Variable(tf.truncated_normal(shape=(5,5,1,6),mean=mu,stddev=sigma))
  conv1_b = tf.Variable(tf.zeros(6))
  conv1 = tf.nn.conv2d(x, conv1_W, strides=[1,1,1,1], padding='VALID') + conv1_b
  # RELU- nonlinear function
  conv1 = tf.nn.relu(conv1)
  # MAX POOL input = 28x28x6 output=14x14x6
  conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')


  # Layer 2 ----------
  conv2_W = tf.Variable(tf.truncated_normal(shape=(5,5,6,16),mean=mu,stddev=sigma))
  conv2_b = tf.Variable(tf.zeros(16))
  conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1,1,1,1],padding='VALID')+conv2_b
  # RELU
  conv2 = tf.nn.relu(conv2)
  # MAX POOL input = 10x10x16 output=5x5x16
  conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  

  # FLATTEN input = 5x5x16 output=400
  fc0 = flatten(conv2)

  # Layer 3: Fully Connected Layer Input=400. Output=120
  fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120),mean=mu,stddev=sigma))
  fc1_b = tf.Variable(tf.zeros(120))
  fc1 = tf.matmul(fc0, fc1_W)+fc1_b

  # Activation RELu
  fc1 = tf.nn.relu(fc1)
  
  # Dropout
  fc1 = tf.nn.dropout(fc1, 0.7)

  # Layer 4: Fully Connected Layer Input=120. Output=84
  fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu,stddev=sigma))
  fc2_b = tf.Variable(tf.zeros(84))
  fc2 = tf.matmul(fc1, fc2_W) + fc2_b
  #Activation RELu
  fc2 = tf.nn.relu(fc2)
  
  # Dropout
  fc2 = tf.nn.dropout(fc2, 0.7)
  
  # Layer 5: Fully Connected : Input = 84, output=43
  fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
  fc3_b = tf.Variable(tf.zeros(43))
  logits = tf.matmul(fc2, fc3_W) + fc3_b
  
  return logits
  
x = tf.placeholder(tf.float32, (None, 32, 32, 1)) 
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
  num_examples = len(X_data)
  total_accuracy = 0 
  ses = tf.get_default_session()
  for offset in range(0, num_examples, BATCH_SIZE):
    batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
    
    accuracy = ses.run(accuracy_operation, feed_dict={x:batch_x, y:batch_y})
    total_accuracy += (accuracy*len(batch_x))
    return total_accuracy/num_examples

with tf.Session() as ses:
  ses.run(tf.global_variables_initializer())
  num_examples = len(X_train)

  print("Training")
  print()

  for i in range(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    
    for offset in range(0, num_examples, BATCH_SIZE):
      end=offset + BATCH_SIZE
      batch_x , batch_y = X_train[offset:end], y_train[offset:end]
      ses.run(training_operation, feed_dict={x:batch_x, y:batch_y})

    validation_accuracy = evaluate(X_valid, y_valid)
    print('EPOCH {} ...'.format(i+1))
    print('Validation accuracy = {:.3f}'.format(validation_accuracy))
    print()

  try:
    saver
  except NameError:
    saver = tf.train.Saver()
  saver.save(ses, './lenet_traffic')
  print('Model Saved')

