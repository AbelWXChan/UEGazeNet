import tensorflow as tf
import numpy as np
import cv2
import time


def compute_accuracy(v_xs, v_ys):
    global prediction
    start = time.clock()
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    print('------', time.clock()-start, '------')
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


def BN(x_in, n_out):
    axis = [0, 1, 2]
    mean, var = tf.nn.moments(x_in, axis)

    scale = tf.Variable(tf.ones([n_out]))
    offset = tf.Variable(tf.zeros([n_out]))
    epsilon = 0.001
    out = tf.nn.batch_normalization(x_in, mean, var, offset, scale, epsilon)

    return out


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 32*32], name='tf_x')
ys = tf.placeholder(tf.float32, [None, 17], name='tf_y')
keep_prob = tf.placeholder(tf.float32)
# x_image = tf.reshape(xs, [-1, 32, 32, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

# net = tf.layers.conv2d(x_image, 16, 1, 1, 'same', activation=tf.nn.leaky_relu)
# net = BN(net, 16)


# net = tf.layers.conv2d(x_image, 16, 3, 2, 'same', activation=tf.nn.leaky_relu)  # 91%
# net = tf.reshape(net, [-1, 16*16*16])
net = tf.layers.dense(xs, 200, tf.nn.leaky_relu)  # 84%
net = tf.layers.dropout(net, 0.5)
net = tf.layers.dense(net, 200, tf.nn.leaky_relu)
net = tf.layers.dropout(net, 0.5)
prediction = tf.layers.dense(net, 17, tf.nn.softmax)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
# cross_entropy = tf.losses.softmax_cross_entropy(ys, prediction)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
pre_gesture = tf.add(prediction, 0, name='pre_gesture')

sess = tf.Session()
sess_saver = tf.train.Saver()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12

init = tf.global_variables_initializer()
sess.run(init)


def one_hot(a):
    dst = np.zeros([a.shape[0], 17])
    for i in range(a.shape[0]):
        dst[i, int(a[i])] = 1
    return dst


x = np.load('./gesture_data/data.npy')
np.random.shuffle(x)
print(x.shape)
_x = x[:8160, :]

# for j in range(2000):
#     cv2.namedWindow('a', cv2.WINDOW_NORMAL)
#     cv2.imshow('a', train_x[j].reshape([32, 32]))
#     print(train_y[j])
#     cv2.waitKey(0)
test_x = x[8160:, 0:32*32]/255.
test_y = x[8160:, 32*32:]
test_y = one_hot(test_y)
batch_size = 500
for i in range(1000):
    np.random.shuffle(_x)
    train_x = _x[:, 0:32 * 32] / 255.
    train_y = _x[:, 32 * 32:]
    train_y = one_hot(train_y)
    for j in range(14):
        inx = train_x[j*batch_size:(j+1)*batch_size]
        iny = train_y[j*batch_size:(j+1)*batch_size]
        sess.run(train_step, feed_dict={xs: inx, ys: iny})
    print(compute_accuracy(test_x, test_y))
    if i % 100 == 0:
        sess_saver.save(sess, './params/gesture')
        print('------ saved -----')
