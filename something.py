import numpy as np
import tensorflow as tf


def genY(len):
    res = np.zeros(shape = [len,2])
    for i in range(len):
        t = np.random.randint(low = 0, high = 2)
        if (t == 0):
            res[i] = [1,0]
        else:
            res[i] = [0,1]
    return tf.Variable(res)

def compare(y1, y2):
    n = int(y1.shape[0])
    res = np.zeros(shape = [n,1])
    for i in range(n):
        if (y1[i].all() == y2[i].all()):
            res[i] = 1
        else:
            res[i] = 0
    return tf.Variable(res)

sess = tf.InteractiveSession()
y1 = genY(4)
y2 = genY(4)

sess.run(tf.initialize_all_variables())

com = compare(y1.eval(sess),y2.eval(sess))
v = y1.eval(sess)
print(v.shape)
print(y2.eval(sess))
print(com.eval(sess))
