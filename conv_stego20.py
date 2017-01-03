
import glob as gl
import math
import numpy as np
import tensorflow as tf
import time
import sys
import cv2
from sklearn.metrics import confusion_matrix
import time

IMAGE_SIZE = 512
NUM_CHANNELS = 1
PIXEL_DEPTH = 255.
NUM_LABELS = 2

NUM_EPOCHS = 2000

STEGO=50000


FLAGS = tf.app.flags.FLAGS

def read_pgm(filename):
    img1 = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    h, w = img1.shape[:2]
    vis0 = np.zeros((h,w), np.float32)
    vis0[:h, :w] = img1
    return vis0
            

#This method is used to read cover and stego images.
#We consider that stego images can be steganographied with differents keys (in practice this seems to be inefficient...)
def extract_data(indexes):
    cover_dir=FLAGS.cover_dir
    stego_dir=FLAGS.stego_dir

    nbImages = len(indexes)
    data = np.ndarray(
        shape=(nbImages,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS),
        dtype=np.float64)
    labels = []

    for i in xrange(nbImages):
        if indexes[i]<STEGO:
            # Load covers
            filename = cover_dir+str(random_images[indexes[i]]+1)+".pgm"
            #print filename
            image = read_pgm(filename)
            data[i,:,:,0]= (image/PIXEL_DEPTH)-0.5
            labels = labels + [[1.0, 0.0]]
        else:
            # Load stego
            new_index=indexes[i]-STEGO
            filename = stego_dir+str(random_images[new_index]+1)+"_"+str(k_key)+".pgm"
            #print filename
            image = read_pgm(filename)
            data[i,:,:,0]= (image/PIXEL_DEPTH)-0.5
            labels = labels + [[0.0, 1.0]]

    labels = np.array(labels)
                
    return (data, labels)

#Same version but with one key per stego image
def extract_data_single(indexes):
    cover_dir=FLAGS.cover_dir
    stego_dir=FLAGS.stego_dir

    nbImages = len(indexes)
    data = np.ndarray(
        shape=(nbImages,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS),
        dtype=np.float64)
    labels = []
    for i in xrange(nbImages):
        if indexes[i]<STEGO:
            # Load covers
            filename = cover_dir+str(random_images[indexes[i]]+1)+".pgm"
            #print filename
            image = read_pgm(filename)
            data[i,:,:,0]= (image/PIXEL_DEPTH)-0.5
            labels = labels + [[1.0, 0.0]]
        else:
            # Load stego
            new_index=indexes[i]-STEGO
            filename = stego_dir+str(random_images[new_index]+1)+".pgm"
            #print filename
            image = read_pgm(filename)
            data[i,:,:,0]= (image/PIXEL_DEPTH)-0.5
            labels = labels + [[0.0, 1.0]]

    labels = np.array(labels)
    return (data, labels)



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding='SAME')









tf.app.flags.DEFINE_string('cover_dir', '',"""Directory containing cover images.""")
tf.app.flags.DEFINE_string('stego_dir', '',"""directory containing stego images.""")
tf.app.flags.DEFINE_string('stego_test_dir', '',"""directory containing stego images.""")
tf.app.flags.DEFINE_string('network', '',"""Pretrained network.""")

tf.app.flags.DEFINE_string('seed', '',"""Seed.""")
tf.app.flags.DEFINE_string('batch_size', '',"""batch size.""")

network=FLAGS.network
seed=int(FLAGS.seed)


BATCH_SIZE = int(FLAGS.batch_size)

tf.set_random_seed(seed)





sess = tf.InteractiveSession()

# 1 - Define the input x_image 
x = tf.placeholder(tf.float32, shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,1))
x_image = x
# 2 - Define the expected output y_image
y = tf.placeholder(tf.float32, shape=(BATCH_SIZE,2))
y_image = y

#print(x_image.get_shape())
#print(y_image.get_shape())

##########

# A - Definition of the CNN

##########

##### 0 - Paremeter used in the Batch-Normalization
epsilon = 1e-4

##### 1 - High-pass filtering definition (F_0)

F_0=tf.cast(tf.constant([[[[-1/12.]],[[ 2/12.]], [[-2/12.]], [[2/12.]], [[-1/12.]]],[[[2/12.]],[[-6/12.]], [[8/12.]], [[-6/12.]], [[2/12.]]],[[[-2/12.]],[[8/12.]], [[-12/12.]], [[8/12.]], [[-2/12.]]],[[[2/12.]],[[-6/12.]], [[8/12.]], [[-6/12.]], [[2/12.]]],[[[-1/12.]],[[2/12.]], [[-2/12.]], [[2/12.]], [[-1/12.]]]]),"float")


##### 2 - Definition of the first convolutional layer - input image => 1 feature map
# Convolution without F_0 (search for another filter 5x5) - PADDING



z_c = tf.nn.conv2d(tf.cast(x_image, "float"), F_0, strides=[1, 1, 1, 1], padding='SAME')



phase_train = tf.placeholder(tf.bool, name='phase_train')



##### Definition of a function for the following convolution layers - size_in feature maps => size_out feature maps
def my_conv_layer(in1,filter_height,filter_width,size_in,size_out,pooling_size,stride_size,active,fabs,padding_type):
    # Convolution with filter_height x filter_width filters 
    W_conv = weight_variable([filter_height,filter_width,size_in,size_out])
    z_conv=conv2d(in1, W_conv)
    if fabs==1:
        # Absolute activation
        z_conv=tf.abs(z_conv)
    # Batch normalization
 

    beta = tf.Variable(tf.constant(0.0, shape=[size_out]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[size_out]), name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(z_conv, [0, 1, 2]  )
    ema = tf.train.ExponentialMovingAverage(decay=0.1)  #previously 0.3
                        


    
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
             return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))

    BN_conv = tf.nn.batch_normalization(z_conv, mean, var, beta, gamma, epsilon)

    if active==1:
        # TanH activation
        f_conv = tf.nn.tanh(BN_conv)
    else:
        # ReLU activation
        f_conv = tf.nn.relu(BN_conv)
    # Average pooling  - pooling_size x pooling_size - stride_size - PADDING
    out = tf.nn.avg_pool(f_conv,ksize=[1,pooling_size,pooling_size,1], strides=[1,stride_size,stride_size,1], padding=padding_type)
    return out

##### 3 - Definition of the second convolutional layer - 1 feature maps => 8 feature map
f_conv2 = my_conv_layer(z_c,5,5,1,8,5,2,1,1,'SAME')
f_conv2_shape = f_conv2.get_shape().as_list()
print(f_conv2_shape)

##### 4 - Definition of the third convolutional layer - 8 feature maps => 16 feature map
f_conv3 = my_conv_layer(f_conv2,5,5,8,16,5,2,1,0,'SAME')
f_conv3_shape = f_conv3.get_shape().as_list()
print(f_conv3_shape)

##### 5 - Definition of the fourth convolutional layer - 16 feature maps => 32 feature maps
f_conv4 = my_conv_layer(f_conv3,1,1,16,32,5,2,0,0,'SAME')
f_conv4_shape = f_conv4.get_shape().as_list()
print(f_conv4_shape)

##### 6 - Definition of the fifth convolutional layer - 32 feature maps => 64 feature maps
f_conv5 = my_conv_layer(f_conv4,1,1,32,64,5,2,0,0,'SAME')
f_conv5_shape = f_conv5.get_shape().as_list()
print(f_conv5_shape)

##### 7 - Definition of the sixth convolutional layer - 64 feature maps => 128 feature maps
f_conv6 = my_conv_layer(f_conv5,1,1,64,128,5,2,0,0,'SAME')
f_conv6_shape = f_conv6.get_shape().as_list()
print(f_conv6_shape)

##### 8 - Definition of the sixth convolutional layer - 128 feature maps => 256 feature maps
f_conv7 = my_conv_layer(f_conv6,1,1,128,256,16,1,0,0,'VALID')
f_conv7_shape = f_conv7.get_shape().as_list()
print(f_conv7_shape)



##### 9 - Reshaping the final output of the convolutional part 
f_conv_shape = f_conv7.get_shape().as_list()
f_conv = tf.reshape(f_conv7,[f_conv_shape[0],f_conv_shape[1]*f_conv_shape[2]*f_conv_shape[3]])

##### Definition of a function for a fully connected layer - input vector of size_in components => output vector of neurons outputs
def my_fullcon_layer(in1,size_in,neurons):
    # Convolution with filter_height x filter_width filters 
    W_full = weight_variable([size_in,neurons])
    b_full = bias_variable([neurons])
    out = tf.nn.tanh(tf.matmul(in1,W_full)+b_full)
    return out

# Without the hidden layer - input = 128 features - output = 2 softmax neurons outputs
W_fc = weight_variable([256,2])
b_fc = bias_variable([2])
y_pred = tf.nn.softmax(tf.matmul(f_conv,W_fc)+b_fc)

##########

# B - Definition of the variables 

##########

# Definition of the error, optimization method, etc.
cross_entropy = -tf.reduce_sum(y_image*tf.log(y_pred+1e-4))


# Training
train_step = tf.train.MomentumOptimizer(learning_rate=1e-3,momentum=0.9).minimize(cross_entropy)


prediction = y_pred

correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_image,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

rounding = tf.argmax(y_pred,1)
tab = tf.placeholder(tf.float32, [None])
reduce_accuracy = tf.reduce_mean(tab)

##########

# C - Initialization of all variables  

##########

sess.run(tf.initialize_all_variables())

##########




##########

# E - Loading data 

##########


#images are permuted according to the random number generation of the seed
random_images=np.arange(0,10000)
np.random.seed(seed)
np.random.shuffle(random_images)




im_train=random_images[0:5000]
im_test=random_images[5000:10000]






##### 1 - Define training data when no given network
#if network=='':

steg=np.add(im_train,np.ones(im_train.shape,dtype=np.int)*STEGO)
arr_train = np.concatenate((im_train,steg),axis=0)

    
np.random.shuffle(arr_train)
indexes_train = [arr_train[i:i+BATCH_SIZE] for i in xrange(0, len(arr_train), BATCH_SIZE)]
train_size = len(indexes_train)
        

#print arr_train

#print indexes_train




##### 2 - Define testing data




steg=np.add(im_test,np.ones(im_test.shape,dtype=np.int)*STEGO)
arr_test = np.concatenate((im_test,steg),axis=0)
                
            


#test data are shuffled
np.random.seed(seed)
np.random.shuffle(arr_test)
indexes_test = [arr_test[i:i+BATCH_SIZE] for i in xrange(0, len(arr_test), BATCH_SIZE)]
test_size = len(indexes_test)



##########

# F - Training or loading a network

##########
num_epochs = NUM_EPOCHS
saver = tf.train.Saver(max_to_keep=1000)
##### 1 - Train a network
key=np.arange(1,3)

if network=='':
    print("training a network")
    start_time = time.time()
    for ep in xrange(num_epochs):
        np.random.shuffle(key)
        k_key=key[0]
        for step in xrange(train_size-1):

                batch_index = step 
                batch_data, batch_labels = extract_data_single(indexes_train[batch_index])

            
                train_step.run(session=sess, feed_dict={ x:batch_data, y:batch_labels, phase_train: True })

                if step%40 == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    pred_test_index = step % test_size
                    pred_test_data, pred_test_labels = extract_data_single(indexes_test[pred_test_index])
                    
                    print("step %d (epoch %d), %.1f ms, showing prediction"%(step,ep,1000*elapsed_time))
                    train_accuracy = accuracy.eval(session=sess, feed_dict={ x:batch_data, y:batch_labels, phase_train: True })
                    
                    print("Train accuracy - batch "+str(batch_index))
                    print(train_accuracy)
                    
                    test_accuracy = accuracy.eval(session=sess, feed_dict={ x:pred_test_data, y:pred_test_labels, phase_train: False})
                    print("Test accuracy - batch "+str(pred_test_index))
                    print(test_accuracy)

                if step==train_size-1-1:
                    global_test_predlabels = []
                    global_test_truelabels = []
                    gtest_accuracy = np.zeros(shape=(test_size), dtype=np.float32)

                    ##train accuracy only to compute update of batch normalization
                    train_accuracy = accuracy.eval(session=sess, feed_dict={ x:batch_data, y:batch_labels, phase_train: True })

                    for global_test_index in xrange(test_size-1):
                        gtest_data, gtest_labels = extract_data_single(indexes_test[global_test_index])
                        batch_accuracy = accuracy.eval(session=sess, feed_dict={ x:gtest_data, y:gtest_labels, phase_train: False})
                        gtest_accuracy[global_test_index] = batch_accuracy
                        print("Global accuracy batch %d = %.3f"%(global_test_index,gtest_accuracy[global_test_index]))
                        gtest_predlabels = rounding.eval(session=sess, feed_dict={ x:gtest_data, phase_train: False})
                        global_test_predlabels = np.concatenate((global_test_predlabels,gtest_predlabels),axis=0)
                        gtest_truelabels = np.argmax(gtest_labels,1)
                        global_test_truelabels = np.concatenate((global_test_truelabels,gtest_truelabels),axis=0)
                        
                    
                    global_accuracy = reduce_accuracy.eval(session=sess, feed_dict={ tab:gtest_accuracy })
                    print("Global Test accuracy")
                    print(global_accuracy)
                    print("Confusion_matrix")
                    print confusion_matrix(global_test_predlabels,global_test_truelabels)

                    np.random.shuffle(arr_train)
                    indexes_train = [arr_train[i:i+BATCH_SIZE] for i in xrange(0, len(arr_train), BATCH_SIZE)]
                    train_size = len(indexes_train)
                    print("SHUFFLE")
                    
                    saver.save(sess, "my-model20", global_step=ep) 
                    
##### 2 - Load a network
else:
    print("loading a network")
    saver.restore(sess, network)

    global_test_predlabels = []
    global_test_truelabels = []
    gtest_accuracy = np.ndarray(shape=(test_size), dtype=np.float32)
    for global_test_index in xrange(test_size-1):
        gtest_data, gtest_labels = extract_data_single(indexes_test[global_test_index])
        #print gtest_labels
        batch_accuracy = accuracy.eval(session=sess, feed_dict={ x:gtest_data, y:gtest_labels, phase_train.name: False})
        gtest_accuracy[global_test_index] = batch_accuracy
        print("Global accuracy batch %d = %.2f"%(global_test_index,gtest_accuracy[global_test_index]))
        gtest_predlabels = rounding.eval(session=sess, feed_dict={ x:gtest_data, phase_train.name: False})
        #print gtest_predlabels
        global_test_predlabels = np.concatenate((global_test_predlabels,gtest_predlabels),axis=0)
        gtest_truelabels = np.argmax(gtest_labels,1)
        global_test_truelabels = np.concatenate((global_test_truelabels,gtest_truelabels),axis=0)
        
    global_accuracy = reduce_accuracy.eval(session=sess, feed_dict={ tab:gtest_accuracy })
    print("Global Test accuracy")
    print(global_accuracy)
    print("Confusion_matrix")
    print confusion_matrix(global_test_predlabels,global_test_truelabels)


                                                                        
