# -*- coding: utf-8 -*-
# @Time    : 2019/4/1
# @Site    : 测试模型_UT Mult views
# @File    : UT.py
# @Software: PyCharm

import tensorflow as tf
import UT_mult_views
import numpy as np
import utils
import os
import cv2
import math
import MPIIGaze
from sklearn.externals import joblib
# """ Build training (GPU：1) """
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class ModelsUT(object):

    def __init__(self):
        self._data_sources = UT_mult_views.UT()
        self._count_data = utils.count_data
        self._batch_size = 32
        self._learn_rate = 0.001
        self._is_save = True
        self._is_load = None
        self._save_path = './params/landmarks_test'
        self._load_path = './params/landmarks_test'
        self._test_ = None

    """ define the network. """

    def stats_graph(self, graph):
        flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

    def BN(self, x_in, n_out):
        axis = [0, 1, 2]
        mean, var = tf.nn.moments(x_in, axis)

        scale = tf.Variable(tf.ones([n_out]))
        offset = tf.Variable(tf.zeros([n_out]))
        epsilon = 0.001
        out = tf.nn.batch_normalization(x_in, mean, var, offset, scale, epsilon)

        return out

    def bottleneck(self, x_in, h_out, n_out):
        n_in = x_in.get_shape()[-1]
        stride = 1 if n_in == n_out else 2

        h = tf.layers.conv2d(x_in, h_out, 3, stride, 'same')
        h = self.BN(h, h_out)
        h = tf.nn.relu(h)
        # h = tf.layers.conv2d(h, h_out, 3, 1, 'same')
        # h = self.BN(h, h_out)
        # h = tf.nn.relu(h)
        h = tf.layers.conv2d(h, n_out, 3, 1, 'same')
        h = self.BN(h, n_out)
        h = tf.nn.relu(h)

        if n_in != n_out:
            shortcut = tf.layers.conv2d(x_in, n_out, 1, stride, 'same')
            shortcut = self.BN(shortcut, n_out)
        else:
            shortcut = x_in

        return tf.nn.relu(shortcut + h)

    def block(self, x_in, n_out, n):
        h_out = n_out // 4
        out = self.bottleneck(x_in, h_out, n_out)

        for i in range(1, n):
            out = self.bottleneck(out, h_out, n_out)

        return out

    """ UEGazeNet* """

    def builds_UEGazeNet_Direct_model(self):
        test_ = MPIIGaze.MPIIGAZE()
        self._is_save = True
        self._is_load = None
        self._save_path = './params/UEGazeNet_Direct_UT'
        self._load_path = './params/UEGazeNet_Direct_UT'
        self._batch_size = 256
        self._learn_rate = 0.001

        ldmks_graph = tf.Graph()
        with ldmks_graph.as_default():

            tf_x = tf.placeholder(tf.float32, [None, 36, 60, 1], name="x_ldmks")
            tf_y_gaze = tf.placeholder(tf.float32, [None, 2], name='y_angle')
            tf_y_head = tf.placeholder(tf.float32, [None, 2], name='x_head')
            train_x = tf.placeholder(tf.float32, [None, 1], name='x_gaze')
            train_y = tf.placeholder(tf.float32, [None, 1], name='y_gaze')
            train_z = tf.placeholder(tf.float32, [None, 1], name='z_gaze')

            # network
            net = tf.layers.conv2d(tf_x, 24, 3, 1, 'same', activation=tf.nn.relu)

            extra_net_1 = tf.layers.conv2d(net, 24, 3, 1, activation=tf.nn.relu)

            net_2 = self.block(net, 24, 1)
            net_2 = tf.layers.conv2d(net_2, 24, 3, 1, activation=tf.nn.relu)
            extra_net_2 = tf.layers.conv2d(net_2 + extra_net_1, 24, 3, 1, activation=tf.nn.relu)

            net_3 = self.block(net_2, 24, 1)
            net_3 = tf.layers.conv2d(net_3, 24, 3, 1, activation=tf.nn.relu)
            extra_net_3 = tf.layers.conv2d(net_3 + extra_net_2, 48, 3, 2, 'same', activation=tf.nn.relu)

            net_4 = self.block(net_3, 48, 1)
            extra_net_4 = tf.layers.conv2d(net_4 + extra_net_3, 50, 3, 1, activation=tf.nn.relu)

            net_5 = self.block(net_4, 48, 1)
            net_5 = tf.layers.conv2d(net_5, 48, 3, 1, activation=tf.nn.relu)
            extra_net_5 = tf.layers.conv2d(net_5 + extra_net_4, 48, 3, 2, 'same', activation=tf.nn.relu)

            net_t = tf.reshape(extra_net_5, [-1, 7 * 13 * 48])
            net_t = tf.layers.dense(net_t, 250, tf.nn.relu)

            net_p = tf.reshape(net_5, [-1, 14 * 26 * 48])
            net_p = tf.layers.dense(net_p, 250, tf.nn.relu)
            net_g = tf.concat([net_p, net_t], 1)
            # net_g = tf.concat([net_g, tf_y_head], 1)
            pre_gaze = tf.layers.dense(net_g, 2)

            """ loss """
            def cal_loss(tf_a, pre_b):
                return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_a, pre_b)), 1)))

            loss = cal_loss(tf_y_gaze, pre_gaze)

            loss = loss / np.pi * 180.
            train_op_ldmks = tf.train.AdamOptimizer(self._learn_rate).minimize(loss)

            loss = tf.add(loss, 0., name='loss')
            pre_gaze = tf.add(pre_gaze, 0, name='pre_gaze')

            self.stats_graph(ldmks_graph)

        # self._count_data = 20188//2
        # subscript = np.random.randint(0, 2304144 - 1, [64000])
        subscript = np.loadtxt('./params/subscript.txt')
        with tf.Session(graph=ldmks_graph) as sess_ldmks:
            sess_ldmks.run(tf.global_variables_initializer())
            saver_ldmks_mc = tf.train.Saver()
            if self._is_load:
                saver_ldmks_mc.restore(sess_ldmks, self._load_path)

            for Step in range(15):

                for step in range(64000 // self._batch_size):

                    input_data, gaze, three, head, head_t = self._data_sources.data_sources(subscript, step,
                                                                                            self._batch_size)
                    in_x = input_data.reshape([self._batch_size, 36, 60, 1]) / 255.

                    sess_ldmks.run(train_op_ldmks,
                                   feed_dict={tf_x: in_x, tf_y_gaze: gaze, tf_y_head: head,
                                              train_x: np.reshape(three[:, 0], [self._batch_size, 1]),
                                              train_y: np.reshape(three[:, 1], [self._batch_size, 1]),
                                              train_z: np.reshape(three[:, 2], [self._batch_size, 1])})

                    loss_1 = sess_ldmks.run(loss_sub, feed_dict={tf_x: in_x, tf_y_gaze: gaze, tf_y_head: head,
                                                                 train_x: np.reshape(three[:, 0],
                                                                                     [self._batch_size, 1]),
                                                                 train_y: np.reshape(three[:, 1],
                                                                                     [self._batch_size, 1]),
                                                                 train_z: np.reshape(three[:, 2],
                                                                                     [self._batch_size, 1])})

                    pre1 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: in_x, tf_y_gaze: gaze, tf_y_head: head})

                    def two2three(n, m):
                        x = -math.cos(n) * math.sin(m)
                        y = -math.sin(n)
                        z = -math.cos(n) * math.cos(m)
                        return x, y, z

                    def cal_acc(cal_pre, cal_y):
                        res = 0

                        for ii in range(self._batch_size):
                            a_l, a_n, a_m = two2three(cal_pre[ii, 0], cal_pre[ii, 1])
                            b_l, b_n, b_m = two2three(cal_y[ii, 0], cal_y[ii, 1])
                            temp = math.acos((a_l * b_l + a_n * b_n + a_m * b_m) / (
                                    math.sqrt(a_l * a_l + a_n * a_n + a_m * a_m) * math.sqrt(
                                b_l * b_l + b_n * b_n + b_m * b_m))) / math.pi * 180
                            res += abs(temp)
                        return res / self._batch_size

                    subscript1 = np.random.randint(0, 213659 - 1, [self._batch_size])
                    right_img, right_gaze = test_.data_sources(subscript1)
                    ERROR = 0
                    for i in range(self._batch_size):
                        test_img = cv2.imread(right_img[i], 0)
                        test_img = np.array(cv2.resize(test_img, (60, 36)))
                        test_gaze = np.loadtxt(right_gaze[i])
                        test_head = cv2.Rodrigues(test_gaze[:3])[0][:, 2]
                        test_pose = [[math.asin(-test_head[1]), math.atan2(-test_head[0], -test_head[2])]]
                        test_img = utils.pro_data(test_img)
                        test_img = utils.Sharpen(test_img)
                        test_img = test_img.reshape([1, 36, 60, 1]) / 255.

                        gaze1 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_img, tf_y_head: test_pose})[0]
                        test_x = test_gaze[0]
                        test_y = test_gaze[1]
                        test_z = test_gaze[2]
                        x, y, z = two2three(gaze1[0], gaze1[1])

                        def cal_error(a_l, a_n, a_m, b_l, b_n, b_m):
                            # a = cal_angle(a_l, a_n, a_m)
                            # b = cal_angle(b_l, b_n, b_m)
                            res = math.acos((a_l * b_l + a_n * b_n + a_m * b_m) / (
                                    math.sqrt(a_l * a_l + a_n * a_n + a_m * a_m) * math.sqrt(
                                b_l * b_l + b_n * b_n + b_m * b_m))) / math.pi * 180
                            return abs(res)

                        error = cal_error(test_x, test_y, test_z, x, y, z)
                        ERROR += error

                    print(step, "train: ", loss_1, cal_acc(pre1, gaze), "\ttest: ",
                          ERROR / (self._batch_size + 1))
                    # print("-----", pre1[:2], "\t", gaze[:2], "----")

                    if self._is_save and step % 50 == 0:
                        print(Step, self._learn_rate, "-----------------  Have been save. ----------------")
                        saver_ldmks_mc.save(sess_ldmks, self._save_path)
                if Step+1 % 2 == 0:
                    self._learn_rate *= 0.1

    """ GazeNet """

    def builds_GazeNet_model(self):
        test_ = MPIIGaze.MPIIGAZE()
        self._is_save = True
        self._is_load = None
        self._save_path = './params/GazeNet_Direct_UT'
        self._load_path = './params/GazeNet_Direct_UT'
        self._batch_size = 256
        self._learn_rate = 0.001

        ldmks_graph = tf.Graph()
        with ldmks_graph.as_default():

            tf_x = tf.placeholder(tf.float32, [None, 36, 60, 1], name="x_ldmks")
            tf_y_gaze = tf.placeholder(tf.float32, [None, 2], name='y_angle')
            tf_y_head = tf.placeholder(tf.float32, [None, 2], name='x_head')
            train_x = tf.placeholder(tf.float32, [None, 1], name='x_gaze')
            train_y = tf.placeholder(tf.float32, [None, 1], name='y_gaze')
            train_z = tf.placeholder(tf.float32, [None, 1], name='z_gaze')

            # network
            conv1 = tf.layers.conv2d(tf_x, 20, 3, 1, activation=tf.nn.relu)
            conv1_1 = tf.layers.conv2d(conv1, 20, 3, 1, activation=tf.nn.relu)
            conv1_2 = tf.layers.conv2d(conv1_1, 20, 3, 1, 'same', activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2)

            conv2_1 = tf.layers.conv2d(pool1, 20, 3, 1, 'same', activation=tf.nn.relu)
            conv2_2 = tf.layers.conv2d(conv2_1, 20, 3, 1, 'same', activation=tf.nn.relu)

            conv2_4 = tf.layers.conv2d(conv2_2, 20, 3, 1, activation=tf.nn.relu)

            conv3_1 = tf.layers.conv2d(conv2_4, 50, 3, 1, activation=tf.nn.relu)
            conv3_2 = tf.layers.conv2d(conv3_1, 50, 3, 1, 'same', activation=tf.nn.relu)
            conv3_3 = tf.layers.conv2d(conv3_2, 50, 3, 1, 'same', activation=tf.nn.relu)

            pool2 = tf.layers.max_pooling2d(conv3_3, 2, 2)

            conv4_1 = tf.layers.conv2d(pool2, 50, 3, 1, 'same', activation=tf.nn.relu)
            conv4_2 = tf.layers.conv2d(conv4_1, 50, 3, 1, 'same', activation=tf.nn.relu)
            conv4_3 = tf.layers.conv2d(conv4_2, 50, 3, 1, 'same', activation=tf.nn.relu)

            fc1 = tf.reshape(conv4_3, [-1, 50 * 6 * 12])
            fc2 = tf.layers.dense(fc1, 500, tf.nn.relu)
            # fc2 = tf.concat([fc2, tf_y_head], 1)
            fc2 = tf.layers.dense(fc2, 500, tf.nn.relu)
            pre_gaze = tf.layers.dense(fc2, 2)

            """ sub_error """

            def cal_sub_loss(tf_a, pre_b):
                return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_a, pre_b)), 1)))

            loss_sub = cal_sub_loss(tf_y_gaze, pre_gaze)

            loss_sub = loss_sub / np.pi * 180.
            loss = loss_sub
            train_op_ldmks = tf.train.AdamOptimizer(self._learn_rate, beta2=0.95).minimize(loss)

            loss_sub = tf.add(loss_sub, 0., name='loss_sub')
            pre_gaze = tf.add(pre_gaze, 0, name='pre_gaze')

            # self.stats_graph(ldmks_graph)

        # self._count_data = 20188//2
        # subscript = np.random.randint(0, 2304144 - 1, [64000])
        # np.savetxt('./params/subscript.txt', subscript)
        subscript = np.loadtxt('./params/subscript.txt')
        with tf.Session(graph=ldmks_graph) as sess_ldmks:
            sess_ldmks.run(tf.global_variables_initializer())
            saver_ldmks_mc = tf.train.Saver()
            if self._is_load:
                saver_ldmks_mc.restore(sess_ldmks, self._load_path)

            for Step in range(25):

                for step in range(64000 // self._batch_size):

                    input_data, gaze, three, head, head_t = self._data_sources.data_sources(subscript, step, self._batch_size)
                    in_x = input_data.reshape([self._batch_size, 36, 60, 1]) / 255.

                    sess_ldmks.run(train_op_ldmks,
                                   feed_dict={tf_x: in_x, tf_y_gaze: gaze, tf_y_head: head,
                                              train_x: np.reshape(three[:, 0], [self._batch_size, 1]),
                                              train_y: np.reshape(three[:, 1], [self._batch_size, 1]),
                                              train_z: np.reshape(three[:, 2], [self._batch_size, 1])})

                    loss_1 = sess_ldmks.run(loss_sub, feed_dict={tf_x: in_x, tf_y_gaze: gaze, tf_y_head: head,
                                                                 train_x: np.reshape(three[:, 0],
                                                                                     [self._batch_size, 1]),
                                                                 train_y: np.reshape(three[:, 1],
                                                                                     [self._batch_size, 1]),
                                                                 train_z: np.reshape(three[:, 2],
                                                                                     [self._batch_size, 1])})

                    pre1 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: in_x, tf_y_gaze: gaze, tf_y_head: head})

                    def two2three(n, m):
                        x = -math.cos(n) * math.sin(m)
                        y = -math.sin(n)
                        z = -math.cos(n) * math.cos(m)
                        return x, y, z

                    def cal_acc(cal_pre, cal_y):
                        res = 0

                        for ii in range(self._batch_size):
                            a_l, a_n, a_m = two2three(cal_pre[ii, 0], cal_pre[ii, 1])
                            b_l, b_n, b_m = two2three(cal_y[ii, 0], cal_y[ii, 1])
                            temp = math.acos((a_l * b_l + a_n * b_n + a_m * b_m) / (
                                    math.sqrt(a_l * a_l + a_n * a_n + a_m * a_m) * math.sqrt(
                                b_l * b_l + b_n * b_n + b_m * b_m))) / math.pi * 180
                            res += abs(temp)
                        return res / self._batch_size

                    subscript1 = np.random.randint(0, 213659 - 1, [self._batch_size])
                    right_img, right_gaze = test_.data_sources(subscript1)
                    ERROR = 0
                    for i in range(self._batch_size):
                        test_img = cv2.imread(right_img[i], 0)
                        test_img = np.array(cv2.resize(test_img, (60, 36)))
                        test_gaze = np.loadtxt(right_gaze[i])
                        test_head = cv2.Rodrigues(test_gaze[:3])[0][:, 2]
                        test_pose = [[math.asin(-test_head[1]), math.atan2(-test_head[0], -test_head[2])]]
                        test_img = utils.pro_data(test_img)
                        test_img = utils.Sharpen(test_img)
                        test_img = test_img.reshape([1, 36, 60, 1]) / 255.

                        gaze1 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_img, tf_y_head: test_pose})[0]
                        test_x = test_gaze[0]
                        test_y = test_gaze[1]
                        test_z = test_gaze[2]
                        x, y, z = two2three(gaze1[0], gaze1[1])

                        def cal_error(a_l, a_n, a_m, b_l, b_n, b_m):
                            # a = cal_angle(a_l, a_n, a_m)
                            # b = cal_angle(b_l, b_n, b_m)
                            res = math.acos((a_l * b_l + a_n * b_n + a_m * b_m) / (
                                    math.sqrt(a_l * a_l + a_n * a_n + a_m * a_m) * math.sqrt(
                                b_l * b_l + b_n * b_n + b_m * b_m))) / math.pi * 180
                            return abs(res)

                        error = cal_error(test_x, test_y, test_z, x, y, z)
                        ERROR += error

                    print(step, "train: ", loss_1, cal_acc(pre1, gaze), "\ttest: ",
                          ERROR / (self._batch_size + 1))
                    # print("-----", pre1[:2], "\t", gaze[:2], "----")

                    if self._is_save and step % 50 == 0:
                        print(Step, self._learn_rate, "-----------------  Have been save. ----------------")
                        saver_ldmks_mc.save(sess_ldmks, self._save_path)
                if Step < 25 and Step % 2 == 0:
                    self._learn_rate *= 0.1

    """ ResNet* """

    def builds_ResNet_Direct_model(self):
        test_ = MPIIGaze.MPIIGAZE()
        self._is_save = True
        self._is_load = None
        self._save_path = './params/ResNet_Direct_UT'
        self._load_path = './params/ResNet_Direct_UT'
        self._batch_size = 256
        self._learn_rate = 0.001

        ldmks_graph = tf.Graph()
        with ldmks_graph.as_default():

            tf_x = tf.placeholder(tf.float32, [None, 36, 60, 1], name="x_ldmks")
            tf_y_gaze = tf.placeholder(tf.float32, [None, 2], name='y_angle')
            tf_y_head = tf.placeholder(tf.float32, [None, 2], name='x_head')
            train_x = tf.placeholder(tf.float32, [None, 1], name='x_gaze')
            train_y = tf.placeholder(tf.float32, [None, 1], name='y_gaze')
            train_z = tf.placeholder(tf.float32, [None, 1], name='z_gaze')

            # network
            net = tf.layers.conv2d(tf_x, 24, 3, 1, 'same', activation=tf.nn.relu)  # [None, 36, 60, 1]
            # net = tf.layers.dropout(net, 0.3)
            net = tf.nn.relu(self.BN(net, 24))

            extra_net_1 = tf.layers.conv2d(net, 24, 3, 1, 'same', activation=tf.nn.relu)  # [None, 36, 60, 1]

            net_2 = self.block(net, 24, 2)  # [None, 36, 60, 1]

            net_3 = self.block(net_2, 24, 2)  # [None, 36, 60, 1]

            net_4 = self.block(net_3, 48, 2)  # [None, 18, 30, 1]

            net_5 = self.block(net_4, 48, 2)

            net_p = tf.reshape(net_5, [-1, 18 * 30 * 48])
            net_p = tf.layers.dense(net_p, 500, tf.nn.relu)
            pre_gaze = tf.layers.dense(net_p, 2)

            """ sub_error """

            def cal_sub_loss(tf_a, pre_b):
                return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_a, pre_b)), 1)))

            loss_sub = cal_sub_loss(tf_y_gaze, pre_gaze)

            loss_sub = loss_sub / 3.1415926 * 180.
            loss = loss_sub  # + loss_angle
            train_op_ldmks = tf.train.AdamOptimizer(self._learn_rate).minimize(loss)

            loss_sub = tf.add(loss_sub, 0., name='loss_sub')
            pre_gaze = tf.add(pre_gaze, 0, name='pre_gaze')

            # self.stats_graph(ldmks_graph)

        test_ = MPIIGaze.MPIIGAZE()
        subscript = np.random.randint(0, 2304144 - 1, [64000])
        with tf.Session(graph=ldmks_graph) as sess_ldmks:
            sess_ldmks.run(tf.global_variables_initializer())
            saver_ldmks_mc = tf.train.Saver()
            if self._is_load:
                saver_ldmks_mc.restore(sess_ldmks, self._load_path)

            for Step in range(25):

                for step in range(64000 // self._batch_size):

                    input_data, gaze, three, head, head_t = self._data_sources.data_sources(subscript, step,
                                                                                            self._batch_size)
                    in_x = input_data.reshape([self._batch_size, 36, 60, 1]) / 255.

                    sess_ldmks.run(train_op_ldmks,
                                   feed_dict={tf_x: in_x, tf_y_gaze: gaze, tf_y_head: head,
                                              train_x: np.reshape(three[:, 0], [self._batch_size, 1]),
                                              train_y: np.reshape(three[:, 1], [self._batch_size, 1]),
                                              train_z: np.reshape(three[:, 2], [self._batch_size, 1])})

                    loss_1 = sess_ldmks.run(loss_sub, feed_dict={tf_x: in_x, tf_y_gaze: gaze, tf_y_head: head,
                                                                 train_x: np.reshape(three[:, 0],
                                                                                     [self._batch_size, 1]),
                                                                 train_y: np.reshape(three[:, 1],
                                                                                     [self._batch_size, 1]),
                                                                 train_z: np.reshape(three[:, 2],
                                                                                     [self._batch_size, 1])})

                    pre1 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: in_x, tf_y_gaze: gaze, tf_y_head: head})

                    def two2three(n, m):
                        x = -math.cos(n) * math.sin(m)
                        y = -math.sin(n)
                        z = -math.cos(n) * math.cos(m)
                        return x, y, z

                    def cal_acc(cal_pre, cal_y):
                        res = 0

                        for ii in range(self._batch_size):
                            a_l, a_n, a_m = two2three(cal_pre[ii, 0], cal_pre[ii, 1])
                            b_l, b_n, b_m = two2three(cal_y[ii, 0], cal_y[ii, 1])
                            temp = math.acos((a_l * b_l + a_n * b_n + a_m * b_m) / (
                                    math.sqrt(a_l * a_l + a_n * a_n + a_m * a_m) * math.sqrt(
                                b_l * b_l + b_n * b_n + b_m * b_m))) / math.pi * 180
                            res += abs(temp)
                        return res / self._batch_size

                    subscript1 = np.random.randint(0, 213659 - 1, [self._batch_size])
                    right_img, right_gaze = test_.data_sources(subscript1)
                    ERROR = 0
                    for i in range(self._batch_size):
                        test_img = cv2.imread(right_img[i], 0)
                        test_img = np.array(cv2.resize(test_img, (60, 36)))
                        test_gaze = np.loadtxt(right_gaze[i])
                        test_head = cv2.Rodrigues(test_gaze[:3])[0][:, 2]
                        test_pose = [[math.asin(-test_head[1]), math.atan2(-test_head[0], -test_head[2])]]
                        test_img = utils.pro_data(test_img)
                        test_img = utils.Sharpen(test_img)
                        test_img = test_img.reshape([1, 36, 60, 1]) / 255.

                        gaze1 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_img, tf_y_head: test_pose})[0]
                        test_x = test_gaze[0]
                        test_y = test_gaze[1]
                        test_z = test_gaze[2]
                        x, y, z = two2three(gaze1[0], gaze1[1])

                        def cal_error(a_l, a_n, a_m, b_l, b_n, b_m):
                            # a = cal_angle(a_l, a_n, a_m)
                            # b = cal_angle(b_l, b_n, b_m)
                            res = math.acos((a_l * b_l + a_n * b_n + a_m * b_m) / (
                                    math.sqrt(a_l * a_l + a_n * a_n + a_m * a_m) * math.sqrt(
                                b_l * b_l + b_n * b_n + b_m * b_m))) / math.pi * 180
                            return abs(res)

                        error = cal_error(test_x, test_y, test_z, x, y, z)
                        ERROR += error

                    print(step, "train: ", loss_1, cal_acc(pre1, gaze), "\ttest: ",
                          ERROR / (self._batch_size + 1))
                    # print("-----", pre1[:2], "\t", gaze[:2], "----")

                    if self._is_save and step % 50 == 0:
                        print(Step, self._learn_rate, "-----------------  Have been save. ----------------")
                        saver_ldmks_mc.save(sess_ldmks, self._save_path)
                if Step < 25 and Step % 5 == 0:
                    self._learn_rate *= 0.1

    def builds_KNN_model(self):
        KNN_model = joblib.load("./ml/params/KNN.model")
        subscript = np.random.randint(0, 2304144 - 1, [64000])
        input_data, gaze, three = self._data_sources.data_sources(subscript, 0, 64000, 15, 9)
        in_x = input_data.reshape([-1, 15 * 9]) / 255.
        KNN_model.fit(in_x, gaze)
        joblib.dump(KNN_model, "./params/KNN_UT.model")

    def builds_RF_model(self):
        RF_model = joblib.load("./ml/params/RF.model")
        subscript = np.random.randint(0, 2304144 - 1, [64000])
        input_data, gaze, three = self._data_sources.data_sources(subscript, 0, 64000, 15, 9)
        in_x = input_data.reshape([-1, 15 * 9]) / 255.
        RF_model.fit(in_x, gaze)
        joblib.dump(RF_model, "./params/RF_UT.model")

    def train_model(self, name):
        if name == 'KNN':
            self.builds_KNN_model()
        elif name == 'RF':
            self.builds_RF_model()
        elif name == 'GazeNet':
            self.builds_GazeNet_S_model()
        elif name == 'UEGazeNet_D_S':
            self.builds_UEGazeNet_Direct_S_model()
        elif name == 'ResNet_D_S':
            self.builds_ResNet_Direct_S_model()


work = ModelsUT()