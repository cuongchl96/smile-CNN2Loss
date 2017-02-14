import tensorflow as tf
import numpy as np
import util
import cv2

def loadData(dataFolder, N):
    imgs = []
    labels = []
    f = open(dataFolder+"/label.txt")
    for line in f:
        v = int(line[:-1])
        if v== 0:
            labels.append([1.,0.])
        else:
            labels.append([0.,1.])
    f.close()
    for i in range(N):
        imgs.append(cv2.imread(dataFolder+"/"+str(i)+".jpg"))
    return imgs[0:3000],labels[0:3000],imgs[3000:],labels[3000:]

def getBatch(p, imgs,labels):
    M = len(p)
    bimgs = np.zeros([M,90, 90],dtype=np.float32)
    blabel = np.zeros([M,2], dtype= np.float32)
    for i in range(M):
        for x in range(1):
            bimgs[i+x] = util.randSelect(imgs[p[i]], 90, 90).astype(np.float32)
            blabel[i+x] = labels[p[i]]
    return [bimgs, blabel]

def compare(y1, y2):
    n = y1.shape[0]
    res = np.zeros([n,1])
    for i in range(n):
        if (y1[i,0] == y2[i,0]):
            res[i] = 1
        else:
            res[i] = 0
    return res

def weight_variable(shape,name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name)

def bias_variable(shape, name):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial,name)

sess = tf.InteractiveSession()

variables_dict = {
    "W_conv1": weight_variable([11,11,1,32],"W_conv1"),
    "b_conv1": bias_variable([32],"b_conv1"),
    "W_conv2": weight_variable([5, 5, 32, 96],"W_conv2"),
    "b_conv2": bias_variable([96],"b_conv2"),
    "W_conv3": weight_variable([5, 5, 96, 128],"W_conv3"),
    "b_conv3": bias_variable([128],"b_conv3"),
    "W_conv4": weight_variable([5, 5, 128, 96],"W_conv4"),
    "b_conv4": bias_variable([96],"b_conv4"),
    "W_fc1": weight_variable([3456, 160],"W_fc1"),
    "b_fc1": bias_variable([160],"b_fc1")
}

keep_prob = tf.placeholder(tf.float32)

def basicCNN(x_image):
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, variables_dict["W_conv1"], strides=[1, 4, 4, 1], padding='SAME') + variables_dict["b_conv1"])
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_pool1 = tf.nn.local_response_normalization(h_pool1)

    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, variables_dict["W_conv2"], strides=[1, 1, 1, 1], padding='SAME') + variables_dict["b_conv2"])
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_pool2 = tf.nn.local_response_normalization(h_pool2)

    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, variables_dict["W_conv3"], strides=[1, 1, 1, 1], padding='SAME') + variables_dict["b_conv3"])

    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, variables_dict["W_conv4"], strides=[1, 1, 1, 1], padding='SAME') + variables_dict["b_conv4"])

    h_flat = tf.reshape(h_conv4, [-1, 3456])
    h_fc1 = tf.nn.relu(tf.matmul(h_flat, variables_dict ["W_fc1"]) + variables_dict ["b_fc1"])

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    return h_fc1_drop

# Signal 1: Main network
x1 = tf.placeholder(tf.float32, shape=[None, 90, 90])
y1_ = tf.placeholder(tf.float32, shape=[None, 2])
x_image1 = tf.reshape(x1, [-1,90,90,1])
signal1 = basicCNN(x_image1)
f1 = tf.reduce_sum(signal1,1,True)
W_last1 = weight_variable([160, 2],"W_last1")
b_last1 = bias_variable([2],"b_last1")
y_conv1 = tf.matmul(signal1, W_last1) + b_last1


# Signal 2: Supervision network
x2 = tf.placeholder(tf.float32, shape=[None, 90, 90])
y2_ = tf.placeholder(tf.float32, shape=[None, 2])
x_image2 = tf.reshape(x2, [-1,90,90,1])
signal2 = basicCNN(x_image2)
f2 = tf.reduce_sum(signal2,1,True)
W_last2 = weight_variable([160, 2],"W_last2")
b_last2 = bias_variable([2],"b_last2")
y_conv2 = tf.matmul(signal2, W_last2) + b_last2

pro_y = tf.placeholder(tf.float32,shape=[None,1])

# Signal 2's cost function and training
cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv2, y2_))
train_step2 = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9).minimize(cross_entropy2)
correct_prediction2 = tf.equal(tf.argmax(y_conv2,1), tf.argmax(y2_,1))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

# Signal 1's cost function and training
reg_param = 0.01
reg_loss = reg_param * (tf.nn.l2_loss(variables_dict["W_conv1"]) +
                        tf.nn.l2_loss(variables_dict["W_conv2"]) +
                        tf.nn.l2_loss(variables_dict["W_conv3"]) +
                        tf.nn.l2_loss(variables_dict["W_conv4"]) +
                        tf.nn.l2_loss(variables_dict["W_fc1"]) +
                        tf.nn.l2_loss(W_last1))

VeLoss_k = 0.005
sigma = 100
f = tf.sub(f1,f2)
f = tf.mul(f,f)
VeLoss = tf.reduce_mean(0.5 * (pro_y * f)) + tf.reduce_mean(0.5 * (1-pro_y) * tf.maximum(0.,sigma - f))

cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv1, y1_)) + reg_loss + VeLoss * VeLoss_k
train_step1 = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9).minimize(cross_entropy1)
correct_prediction1 = tf.equal(tf.argmax(y_conv1,1), tf.argmax(y1_,1))
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

dataFolder = "train"

trainImg, trainLabel, testImg, testLabel = loadData(dataFolder, 4000)

N = len(trainImg)

batch_size = 256

sess.run(tf.initialize_all_variables())
best_accuracy = 0.


for i in range(500):
    p1 = np.arange(N)
    np.random.shuffle(p1)

    p2 = np.arange(N)
    np.random.shuffle(p2)

    for indData in range(0, N - batch_size + 1, batch_size):
        bimg1, blabel1 = getBatch(p1[indData:indData+batch_size],trainImg, trainLabel)
        bimg2, blabel2 = getBatch(p2[indData:indData+batch_size],trainImg, trainLabel)
        compare_y = compare(blabel1,blabel2)

        f2.eval(feed_dict={x2: bimg2, y2_: blabel2, pro_y: compare_y.astype(np.float32), keep_prob: 0.5})
        train_step1.run(feed_dict={x1: bimg1, y1_: blabel1, pro_y: compare_y.astype(np.float32), x2: bimg2, y2_: blabel2,keep_prob: 0.5})
        #print(VeLoss.eval(sess))

    print("Epoch %d" %i)
    ti, tl = getBatch(range(1000), testImg, testLabel)
    acc1 = accuracy1.eval(feed_dict={x1: ti, y1_: tl, keep_prob: 1.0})
    acc2 = accuracy2.eval(feed_dict={x2: ti, y2_: tl, keep_prob: 1.0})
    if (acc1 > best_accuracy):
        best_accuracy = acc1
    print("test accuracy: %g" %acc1)
    print("Best: %g"%best_accuracy)

