# -*- coding: utf-8 -*-
# @Time    : 2019/5/9
# @Site    : 以UnityEye为训练集的模型
# @File    : UE.py
# @Software: PyCharm

import tensorflow as tf
from data_sources import BaseDateSource
import numpy as np
import utils
import os
import cv2
import math
import MPIIGaze
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

""" Build training (GPU：0) """
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class GAZE(object):

    def __init__(self):
        self._load_gaze_path = './params/UEGazeNet_gaze'
        self._gaze_graph = tf.Graph()
        with self._gaze_graph.as_default():
            self._saver_gaze = tf.train.import_meta_graph('./params/UEGazeNet_gaze.meta')
        self._sess_gaze = tf.Session(graph=self._gaze_graph)
        self._saver_gaze.restore(self._sess_gaze, self._load_gaze_path)
        self.prediction_gaze = self._gaze_graph.get_tensor_by_name('pre_gaze:0')
        self.tf_x = self._gaze_graph.get_tensor_by_name('x_gaze:0')

    def reader(self):
        reader = tf.train.NewCheckpointReader(self._load_gaze_path)

        variables = reader.get_variable_to_shape_map()

        for ele in variables:
            print(ele)

    def get_gaze(self, ldmks):
        in_x = ldmks.reshape(1, 110)
        gaze_pre = self._sess_gaze.run(self.prediction_gaze, feed_dict={self.tf_x: in_x / 60.})
        return gaze_pre


class ModelsUE(object):

    def __init__(self):
        self._data_sources = BaseDateSource()
        self._count_data = utils.count_data
        self._batch_size = 32
        self._learn_rate = 0.0001
        self._is_save = True
        self._is_load = None
        self._save_path = './params/test'
        self._load_path = './params/test'
        self._test_ = None

    """ define the res blocks. """

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

    """ train the gaze model. """

    def builds_UEGazeNet_gaze_model(self):
        self._is_save = True
        self._is_load = None
        self._save_path = './params/UEGazeNet_gaze'
        self._load_path = './params/UEGazeNet_gaze'
        self._batch_size = 256
        self._learn_rate = 0.001
        self._data_sources = BaseDateSource(256, 192, True)
        # self._count_data = 20132

        gaze_graph = tf.Graph()
        with gaze_graph.as_default():

            tf_x_gaze = tf.placeholder(tf.float32, [None, 110], name='x_gaze')
            tf_y_gaze = tf.placeholder(tf.float32, [None, 2], name='y_gaze')
            fc_gaze = tf.layers.dense(tf_x_gaze, 32, activation=tf.nn.relu)
            fc_gaze = tf.layers.dense(fc_gaze, 64, activation=tf.nn.relu)
            fc_gaze = tf.layers.dense(fc_gaze, 128, activation=tf.nn.relu)
            fc_gaze = tf.layers.dropout(fc_gaze, 0.3)
            prediction_gaze = tf.layers.dense(fc_gaze, 2)

            def cal_sub_loss(tf_a, pre_b):
                return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_a, pre_b)), 1)))

            loss_sub = cal_sub_loss(tf_y_gaze, prediction_gaze)

            loss_sub = loss_sub

            train_op_gaze = tf.train.AdamOptimizer(self._learn_rate).minimize(loss_sub, name='train_op_gaze')

            loss_gaze_sub = tf.cast(loss_sub, tf.float32, name='loss_gaze_sub')
            pre_gaze = tf.add(prediction_gaze, 0.0, name='pre_gaze')

        with tf.Session(graph=gaze_graph) as sess_gaze:
            sess_gaze.run(tf.global_variables_initializer())
            savers_gaze = tf.train.Saver()  # define a saver for saving and restoring
            if self._is_load:
                savers_gaze.restore(sess_gaze, self._load_path)

            def test_look_vec(w, h):
                img1 = cv2.imread("./test/144.jpg")
                img1 = cv2.resize(img1, (w, h))
                img2 = cv2.imread("./test/477.jpg")
                img2 = cv2.resize(img2, (w, h))
                img3 = cv2.imread("./test/1077.jpg")
                img3 = cv2.resize(img3, (w, h))

                def two2three(m, n):
                    x = -math.cos(n) * math.sin(m)
                    y = -math.sin(n)
                    z = -math.cos(n) * math.cos(m)
                    return x, y, z

                test_x1 = utils.load_a_json('./test/144', w, h)
                test_x1 = test_x1.reshape(1, 110)
                test_pre1 = sess_gaze.run(prediction_gaze, feed_dict={tf_x_gaze: test_x1 / 60.})[0]
                # x1, y1, z1 = two2three(test_pre1[0][0], test_pre1[0][1])
                _, _, eye_p1 = utils.separate_all(1, test_x1)

                test_x2 = utils.load_a_json('./test/477', w, h)
                test_x2 = test_x2.reshape(1, 110)
                test_pre2 = sess_gaze.run(prediction_gaze, feed_dict={tf_x_gaze: test_x2 / 60.})[0]
                # x2, y2, z2 = two2three(test_pre2[0][0], test_pre2[0][1])
                _, _, eye_p2 = utils.separate_all(1, test_x2)

                test_x3 = utils.load_a_json('./test/1077', w, h)
                test_x3 = test_x3.reshape(1, 110)
                test_pre3 = sess_gaze.run(prediction_gaze, feed_dict={tf_x_gaze: test_x3 / 60.})[0]
                # x3, y3, z3 = two2three(test_pre3[0][0], test_pre3[0][1])
                _, _, eye_p3 = utils.separate_all(1, test_x3)

                eye_c1, _ = utils.get_r_c(eye_p1.reshape((32, 2)))
                eye_c2, _ = utils.get_r_c(eye_p2.reshape((32, 2)))
                eye_c3, _ = utils.get_r_c(eye_p3.reshape((32, 2)))

                img_temp2 = img2.copy()
                img_temp3 = img3.copy()
                img_temp1 = img1.copy()

                def drawline(img, a1, a2, b1, b2):
                    cv2.line(img, (a1, a2), (int(a1 + b1 * 30), int(a2 + b2 * 30)), (0, 0, 255), 1)

                drawline(img_temp1, eye_c1[0], eye_c1[1], test_pre1[0], test_pre1[1])
                cv2.imshow('eye1', img_temp1)
                drawline(img_temp2, eye_c2[0], eye_c2[1], test_pre2[0], test_pre2[1])
                cv2.imshow('eye2', img_temp2)
                drawline(img_temp3, eye_c3[0], eye_c3[1], test_pre3[0], test_pre3[1])
                cv2.imshow('eye3', img_temp3)

                cv2.waitKey(1)

            for Step in range(25):

                self._data_sources.init_num()

                for step in range(self._count_data // self._batch_size):

                    self._data_sources.record_data(self._batch_size)
                    input_data = np.array(self._data_sources.ldmks_data())
                    input_data = input_data.reshape(self._batch_size, 110)
                    gaze = np.array(self._data_sources.look_vec())

                    """ gaze normalized. """
                    gaze_angle = np.zeros([self._batch_size, 2])
                    for i in range(self._batch_size):
                        # modular_length = (gaze[i, 0]**2 + gaze[i, 1]**2 + gaze[i, 2]**2)**0.5
                        modular_length = 1.
                        gaze_angle[i, 0] = math.asin(-gaze[i, 1] / modular_length)
                        gaze_angle[i, 1] = math.atan2(-gaze[i, 0] / modular_length, -gaze[i, 2] / modular_length)

                    in_x = input_data
                    in_y = gaze_angle

                    loss, _ = sess_gaze.run([loss_gaze_sub, train_op_gaze], feed_dict={tf_x_gaze: in_x / 256.,
                                                                                       tf_y_gaze: in_y})

                    pre = sess_gaze.run(pre_gaze, feed_dict={tf_x_gaze: in_x / 256., tf_y_gaze: in_y})

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
                            # a_l = cal_pre[ii][0]
                            # a_n = cal_pre[ii][1]
                            # a_m = cal_pre[ii][2]
                            # b_l = cal_y[ii][0]
                            # b_n = cal_y[ii][1]
                            # b_m = cal_y[ii][2]
                            temp = math.acos((a_l * b_l + a_n * b_n + a_m * b_m) / (
                                    math.sqrt(a_l * a_l + a_n * a_n + a_m * a_m) * math.sqrt(
                                b_l * b_l + b_n * b_n + b_m * b_m))) / math.pi * 180
                            res += abs(temp)
                        return res / self._batch_size

                    if step % 2 == 0:
                        print(step, loss, cal_acc(pre, gaze_angle))
                        print("three: ", pre[0], gaze[0])
                        if self._test_:
                            test_look_vec(60, 36)
                    if step % 30 == 0:
                        if self._is_save:
                            print(Step, " Have been saved. ")
                            savers_gaze.save(sess_gaze, self._save_path)

                if Step < 25 and Step % 2:
                    self._learn_rate *= 0.1

    """ UEGazeNet* """

    def builds_UEGazeNet_Direct_model(self):
        self._data_sources = BaseDateSource(60, 36, None)
        test_ = MPIIGaze.MPIIGAZE()
        self._is_save = True
        self._is_load = None
        self._save_path = './params/UEN_Direct_UE'
        self._load_path = './params/UEN_Direct_UE'
        self._batch_size = 256
        self._learn_rate = 0.001
        # self._count_data = 102014 // 2

        ldmks_graph = tf.Graph()
        with ldmks_graph.as_default():

            tf_x = tf.placeholder(tf.float32, [None, 36, 60, 1], name="x_ldmks")
            tf_y_gaze = tf.placeholder(tf.float32, [None, 2], name='y_angle')
            tf_t = tf.placeholder(tf.float32, [None, 1], name='y_angle_t')
            tf_p = tf.placeholder(tf.float32, [None, 1], name='y_angle_p')

            # network
            net = tf.layers.conv2d(tf_x, 24, 3, 1, 'same', activation=tf.nn.relu)  # [None, 36, 60, 1]

            extra_net_1 = tf.layers.conv2d(net, 24, 3, 1, activation=tf.nn.relu)

            net_2 = self.block(net, 24, 1) 
            net_2 = tf.layers.conv2d(net_2, 24, 3, 1, activation=tf.nn.relu)
            extra_net_2 = tf.layers.conv2d(net_2 + extra_net_1, 24, 3, 1, activation=tf.nn.relu)

            net_3 = self.block(net_2, 24, 1)
            net_3 = tf.layers.conv2d(net_3, 24, 3, 1, activation=tf.nn.relu)
            extra_net_3 = tf.layers.conv2d(net_3 + extra_net_2, 48, 3, 2, 'same', activation=tf.nn.relu)

            net_4 = self.block(net_3, 48, 1)
            extra_net_4 = tf.layers.conv2d(net_4 + extra_net_3, 48, 3, 1, activation=tf.nn.relu)

            net_5 = self.block(net_4, 48, 1) 
            net_5 = tf.layers.conv2d(net_5, 48, 3, 1, activation=tf.nn.relu)
            extra_net_5 = tf.layers.conv2d(net_5 + extra_net_4, 48, 3, 2, 'same', activation=tf.nn.relu)

            net_t = tf.reshape(extra_net_5, [-1, 7 * 13 * 48])
            net_t = tf.layers.dense(net_t, 250, tf.nn.relu)

            net_p = tf.reshape(net_5, [-1, 14 * 26 * 48])
            net_p = tf.layers.dense(net_p, 250, tf.nn.relu)
            net_g = tf.concat([net_p, net_t], 1)
            pre_gaze = tf.layers.dense(net_g, 2)


            """ angle_error """

            def cal_angle_loss(a_l, b_l, a_n, b_n, a_m, b_m):
                temp = tf.acos((a_l * b_l + a_n * b_n + a_m * b_m) / (
                        tf.sqrt(a_l * a_l + a_n * a_n + a_m * a_m) * tf.sqrt(
                    b_l * b_l + b_n * b_n + b_m * b_m))) / np.pi * 180
                return tf.reduce_mean(temp)

            # loss_angle = cal_angle_loss(pre_x, train_x, pre_y, train_y, pre_z, train_z)

            """ sub_error """

            def cal_sub_loss(tf_a, pre_b):
                return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_a, pre_b)), 1)))

            loss_sub = cal_sub_loss(tf_y_gaze,
                                    pre_gaze)  # + cal_sub_loss(tf_t, pre_theta) + cal_sub_loss(tf_p, pre_phi)

            loss_sub = loss_sub / np.pi * 180.
            loss = loss_sub  # + loss_angle
            train_op_ldmks = tf.train.AdamOptimizer(self._learn_rate).minimize(loss)

            loss_sub = tf.add(loss_sub, 0., name='loss_sub')
            # loss_angle = tf.add(loss_angle, 0., name='loss_angle')
            pre_gaze = tf.add(pre_gaze, 0, name='pre_gaze')

            # self.stats_graph(ldmks_graph)

        def test_image_all(w, h):
            img1 = cv2.imread("./test/6660.jpg")
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.imread("./test/12.jpg")
            img2 = cv2.resize(img2, (w, h))
            img3 = cv2.imread("./test/999.jpg")
            img3 = cv2.resize(img3, (w, h))

            test_x1 = utils.pro_data(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
            test_x1 = test_x1.reshape(1, h, w, 1) / 255.
            pre_test1 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_x1})
            pre_gaze1 = pre_test1[0, 0:2]

            test_x2 = utils.pro_data(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)).reshape(1, h, w, 1) / 255.
            pre_test2 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_x2})
            pre_gaze2 = pre_test2[0, 0:2]

            test_x3 = utils.pro_data(cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)).reshape(1, h, w, 1) / 255.
            pre_test3 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_x3})
            pre_gaze3 = pre_test3[0, 0:2]

            img_temp1 = img1.copy()
            img_temp2 = img2.copy()
            img_temp3 = img3.copy()

            cv2.arrowedLine(img_temp1, (128, 96), (int(pre_gaze1[0] * 40 + 128), int(pre_gaze1[1] * 40 + 96)),
                            (0, 255, 255), 2)
            cv2.imshow('eye1', img_temp1)

            cv2.arrowedLine(img_temp2, (128, 96), (int(pre_gaze2[0] * 40 + 128), int(pre_gaze2[1] * 40 + 96)),
                            (0, 255, 255), 2)
            cv2.imshow('eye2', img_temp2)

            cv2.arrowedLine(img_temp3, (128, 96), (int(pre_gaze3[0] * 40 + 128), int(pre_gaze3[1] * 40 + 96)),
                            (0, 255, 255), 2)
            cv2.imshow('eye3', img_temp3)

            # cv2.imshow('eyeb3', pre_mask_contour3)

            # print(step, ldmks_caruncle1[0][0], ldmks_caruncle2[0][0], ldmks_caruncle3[0][0])
            cv2.waitKey(1)

        with tf.Session(graph=ldmks_graph) as sess_ldmks:
            sess_ldmks.run(tf.global_variables_initializer())
            saver_ldmks_mc = tf.train.Saver()
            if self._is_load:
                saver_ldmks_mc.restore(sess_ldmks, self._load_path)

            gaze_angle = np.zeros((self._batch_size, 2))
            for Step in range(15):
                self._data_sources.init_num()

                for step in range(self._count_data // self._batch_size):

                    self._data_sources.record_data(self._batch_size)
                    input_data = np.array(self._data_sources.input_image())
                    gaze = np.array(self._data_sources.look_vec())
                    # print(gaze.shape)
                    for i in range(self._batch_size):
                        # input_data[i] = cv2.resize(input_data[i], (60, 36))
                        gaze_angle[i, 0] = math.asin(-gaze[i, 1])
                        gaze_angle[i, 1] = math.atan2(-gaze[i, 0], -gaze[i, 2])
                    # print(gaze_angle.shape)
                    # np.reshape(gaze_angle[:, 0],

                    in_x = input_data.reshape([self._batch_size, 36, 60, 1]) / 255.

                    sess_ldmks.run(train_op_ldmks, feed_dict={tf_x: in_x, tf_y_gaze: gaze_angle,
                                                              tf_t: np.reshape(gaze_angle[:, 0], [self._batch_size, 1]),
                                                              tf_p: np.reshape(gaze_angle[:, 1], [self._batch_size, 1])
                                                              })

                    loss_ = sess_ldmks.run(loss_sub, feed_dict={tf_x: in_x, tf_y_gaze: gaze_angle,
                                                                tf_t: np.reshape(gaze_angle[:, 0],
                                                                                 [self._batch_size, 1]),
                                                                tf_p: np.reshape(gaze_angle[:, 1],
                                                                                 [self._batch_size, 1]),
                                                                })

                    pre = sess_ldmks.run(pre_gaze, feed_dict={tf_x: in_x, tf_y_gaze: gaze_angle})

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

                    subscript = np.random.randint(0, 213659 - 1, [self._batch_size])
                    right_img, right_gaze = test_.data_sources(subscript)
                    ERROR = 0
                    for i in range(self._batch_size):
                        test_img = cv2.imread(right_img[i], 0)
                        test_img = np.array(cv2.resize(test_img, (60, 36)))
                        test_gaze = np.loadtxt(right_gaze[i])
                        test_img = utils.pro_data(test_img)
                        test_img = utils.Sharpen(test_img)
                        test_img = test_img.reshape([1, 36, 60, 1]) / 255.

                        gaze = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_img})[0]
                        test_x = test_gaze[0]
                        test_y = test_gaze[1]
                        test_z = test_gaze[2]
                        x, y, z = two2three(gaze[0], gaze[1])

                        def cal_error(a_l, a_n, a_m, b_l, b_n, b_m):
                            # a = cal_angle(a_l, a_n, a_m)
                            # b = cal_angle(b_l, b_n, b_m)
                            res = math.acos((a_l * b_l + a_n * b_n + a_m * b_m) / (
                                    math.sqrt(a_l * a_l + a_n * a_n + a_m * a_m) * math.sqrt(
                                b_l * b_l + b_n * b_n + b_m * b_m))) / math.pi * 180
                            return abs(res)

                        error = cal_error(test_x, test_y, test_z, x, y, z)
                        ERROR += error
                    # if step % 5 == 0:
                    print(step, loss_, cal_acc(pre, gaze_angle), ERROR / (self._batch_size + 1))
                    if self._test_:
                        test_image_all(60, 36)
                    if self._is_save and step % 50 == 0:
                        print(Step, self._learn_rate, "-----------------  Have been save. ----------------")
                        saver_ldmks_mc.save(sess_ldmks, self._save_path)
                if Step % 2 == 0:
                    self._learn_rate *= 0.1

    """ train UEGazeNet. """

    def builds_UEGazeNet_model(self):
        self._is_save = True
        self._is_load = None
        self._save_path = './params/UEGazeNet'
        self._load_path = './params/UEGazeNet'
        self._batch_size = 32
        self._learn_rate = 0.001
        self._data_sources = BaseDateSource(256, 192, True)

        # self._count_data = 45000
        ldmks_graph = tf.Graph()
        with ldmks_graph.as_default():

            tf_x = tf.placeholder(tf.float32, [None, 192, 256, 1], name="x_ldmks")
            tf_iris = tf.placeholder(tf.float32, [None, 64], name='tf_mask_iris')
            tf_contour = tf.placeholder(tf.float32, [None, 46], name='tf_mask_contour')
            tf_y = tf.placeholder(tf.float32, [None, 110], name='y_ldmks')

            # network
            net = tf.layers.conv2d(tf_x, 16, 3, 2, 'same', activation=tf.nn.relu)
            net = tf.layers.dropout(net, 0.3)
            net = tf.nn.relu(self.BN(net, 16))
            extra_net_1 = tf.layers.conv2d(net, 32, 3, 2, 'same', activation=tf.nn.relu)

            net_2 = self.block(net, 32, 2)
            extra_net_2 = tf.layers.conv2d(net_2 + extra_net_1, 64, 3, 2, 'same', activation=tf.nn.relu)

            net_3 = self.block(net_2, 64, 2)
            extra_net_3 = tf.layers.conv2d(net_3 + extra_net_2, 128, 3, 2, 'same', activation=tf.nn.relu)

            net_4 = self.block(net_3, 128, 2)
            extra_net_4 = tf.layers.conv2d(net_4 + extra_net_3, 256, 3, 2, 'same', activation=tf.nn.relu)

            net_5 = self.block(net_4, 256, 2)
            extra_net_5 = tf.layers.conv2d(net_5 + extra_net_4, 256, 3, 2, 'same', activation=tf.nn.relu)

            net_c = tf.reshape(extra_net_5, [-1, 3 * 4 * 256])
            net_c = tf.layers.dense(net_c, 1000, tf.nn.relu)
            net_c = tf.layers.dropout(net_c, 0.5)
            pre_contours = tf.layers.dense(net_c, 46, name='pre_contour')

            net_i = tf.reshape(net_5, [-1, 6 * 8 * 256])
            net_i = tf.layers.dense(net_i, 1000, tf.nn.relu)
            net_i = tf.layers.dropout(net_i, 0.5)
            pre_iris = tf.layers.dense(net_i, 64, name='pre_iris')
            pre_ldmks = tf.concat([pre_contours, pre_iris], 1)

            def the_difference(pre, inputs):
                temp = tf.reduce_sum(tf.abs(tf.subtract(pre, inputs)))
                score = (temp + 1.) / (tf.abs(tf.reduce_sum(tf.add(pre, inputs))) + 1.)
                return score

            """ ldmks """
            loss_difference1 = the_difference(pre_ldmks, tf_y)
            loss_difference2 = 0.5 * the_difference(pre_contours, tf_contour) + 0.5 * the_difference(pre_iris, tf_iris)
            loss_ldmks = 0.9 * loss_difference1 + 0.1 * loss_difference2
            train_op_ldmks = tf.train.AdamOptimizer(self._learn_rate).minimize(loss_ldmks)
            loss_ldmks = tf.cast(loss_ldmks, tf.float32, name='loss_ldmks_mc')
            pre_test_mc = tf.add(pre_ldmks, 0, name='pre_ldmks_mc')

            self.stats_graph(ldmks_graph)

        def test_image_all(w, h):
            img1 = cv2.imread("./test/6660.jpg")
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.imread("./test/12.jpg")
            img2 = cv2.resize(img2, (w, h))
            img3 = cv2.imread("./test/999.jpg")
            img3 = cv2.resize(img3, (w, h))

            test_x1 = utils.pro_data(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
            test_x1 = test_x1.reshape(1, h, w, 1) / 255.
            test_pre1 = sess_ldmks.run(pre_ldmks, feed_dict={tf_x: test_x1})
            ldmks_interior_margin1, ldmks_caruncle1, ldmks_iris1 = utils.separate_all(1, test_pre1)

            test_x2 = utils.pro_data(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)).reshape(1, h, w, 1) / 255.
            test_pre2 = sess_ldmks.run(pre_ldmks, feed_dict={tf_x: test_x2})
            ldmks_interior_margin2, ldmks_caruncle2, ldmks_iris2 = utils.separate_all(1, test_pre2)
            test_x3 = utils.pro_data(cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)).reshape(1, h, w, 1) / 255.
            test_pre3 = sess_ldmks.run(pre_ldmks, feed_dict={tf_x: test_x3})
            ldmks_interior_margin3, ldmks_caruncle3, ldmks_iris3 = utils.separate_all(1, test_pre3)
            img_temp2 = utils.hisEqulColor(utils.Sharpen(img2))
            img_temp3 = utils.hisEqulColor(utils.Sharpen(img3))
            img_temp1 = utils.hisEqulColor(utils.Sharpen(img1))

            # cv2.imshow('eyeb1', mask)

            # img = cv2.add(rmask, bmask)

            utils.draw_ldmks(img_temp1, ldmks_interior_margin1, ldmks_caruncle1, ldmks_iris1)
            cv2.imshow('eye1', img_temp1)

            utils.draw_ldmks(img_temp2, ldmks_interior_margin2, ldmks_caruncle2, ldmks_iris2)
            cv2.imshow('eye2', img_temp2)

            utils.draw_ldmks(img_temp3, ldmks_interior_margin3, ldmks_caruncle3, ldmks_iris3)
            cv2.imshow('eye3', img_temp3)

            # cv2.imshow('eyeb3', pre_mask_contour3)

            # print(step, ldmks_caruncle1[0][0], ldmks_caruncle2[0][0], ldmks_caruncle3[0][0])
            cv2.waitKey(1)

        with tf.Session(graph=ldmks_graph) as sess_ldmks:
            sess_ldmks.run(tf.global_variables_initializer())
            saver_ldmks_mc = tf.train.Saver()
            if self._is_load:
                saver_ldmks_mc.restore(sess_ldmks, self._load_path)

            for Step in range(25):
                self._data_sources.init_num()

                for step in range(self._count_data // self._batch_size):

                    self._data_sources.record_data(self._batch_size)
                    input_data = np.array(self._data_sources.input_image())
                    landmarks = np.array(self._data_sources.ldmks_data())
                    landmarks = landmarks.reshape((self._batch_size, 110))

                    in_x = input_data.reshape([self._batch_size, 192, 256, 1]) / 255.
                    in_contour, in_iris = utils.separate_contour_and_iris(self._batch_size, landmarks)

                    in_contour = in_contour.reshape((self._batch_size, 46))
                    in_iris = in_iris.reshape((self._batch_size, 64))

                    loss, _ = sess_ldmks.run([loss_ldmks, train_op_ldmks],
                                             feed_dict={tf_x: in_x, tf_y: landmarks, tf_contour: in_contour,
                                                        tf_iris: in_iris})

                    # acc = compute_accuracy(in_x, landmarks)
                    if step % 5 == 0:
                        print(Step, step, loss)
                    if self._test_:
                        test_image_all(256, 192)
                if self._is_save:
                    print(Step, "------------- Have been save. --------------")
                    saver_ldmks_mc.save(sess_ldmks, self._save_path)
                if Step < 25 and Step % 5:
                    self._learn_rate *= 0.1

    """ GazeNet """

    def builds_GazeNet_model(self):
        self._data_sources = BaseDateSource(60, 36, None)
        test_ = MPIIGaze.MPIIGAZE()
        self._is_save = True
        self._is_load = None
        self._save_path = './params/GazeNet_UE'
        self._load_path = './params/GazeNet_UE'
        self._batch_size = 256
        self._learn_rate = 0.001
        self._count_data = 102014 // 2

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

            loss_sub = loss_sub / 3.1415926 * 180.
            loss = loss_sub
            train_op_ldmks = tf.train.AdamOptimizer(self._learn_rate).minimize(loss)

            loss_sub = tf.add(loss_sub, 0., name='loss_sub')
            pre_gaze = tf.add(pre_gaze, 0, name='pre_gaze')

            # self.stats_graph(ldmks_graph)

        def test_image_all(w, h):
            img1 = cv2.imread("./test/6660.jpg")
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.imread("./test/12.jpg")
            img2 = cv2.resize(img2, (w, h))
            img3 = cv2.imread("./test/999.jpg")
            img3 = cv2.resize(img3, (w, h))

            test_x1 = utils.pro_data(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
            test_x1 = test_x1.reshape(1, h, w, 1) / 255.
            pre_test1 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_x1})
            pre_gaze1 = pre_test1[0, 0:2]

            test_x2 = utils.pro_data(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)).reshape(1, h, w, 1) / 255.
            pre_test2 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_x2})
            pre_gaze2 = pre_test2[0, 0:2]

            test_x3 = utils.pro_data(cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)).reshape(1, h, w, 1) / 255.
            pre_test3 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_x3})
            pre_gaze3 = pre_test3[0, 0:2]

            img_temp1 = img1.copy()
            img_temp2 = img2.copy()
            img_temp3 = img3.copy()

            cv2.arrowedLine(img_temp1, (128, 96), (int(pre_gaze1[0] * 40 + 128), int(pre_gaze1[1] * 40 + 96)),
                            (0, 255, 255), 2)
            cv2.imshow('eye1', img_temp1)

            cv2.arrowedLine(img_temp2, (128, 96), (int(pre_gaze2[0] * 40 + 128), int(pre_gaze2[1] * 40 + 96)),
                            (0, 255, 255), 2)
            cv2.imshow('eye2', img_temp2)

            cv2.arrowedLine(img_temp3, (128, 96), (int(pre_gaze3[0] * 40 + 128), int(pre_gaze3[1] * 40 + 96)),
                            (0, 255, 255), 2)
            cv2.imshow('eye3', img_temp3)

            # cv2.imshow('eyeb3', pre_mask_contour3)

            # print(step, ldmks_caruncle1[0][0], ldmks_caruncle2[0][0], ldmks_caruncle3[0][0])
            cv2.waitKey(1)

        with tf.Session(graph=ldmks_graph) as sess_ldmks:
            sess_ldmks.run(tf.global_variables_initializer())
            saver_ldmks_mc = tf.train.Saver()
            if self._is_load:
                saver_ldmks_mc.restore(sess_ldmks, self._load_path)

            gaze_angle = np.zeros((self._batch_size, 2))
            head_angle = np.zeros((self._batch_size, 2))
            for Step in range(25):
                self._data_sources.init_num()

                for step in range(self._count_data // self._batch_size):

                    self._data_sources.record_data(self._batch_size)
                    input_data = np.array(self._data_sources.input_image())
                    gaze = np.array(self._data_sources.look_vec())
                    head = np.array(self._data_sources.head_vec())
                    for i in range(self._batch_size):
                        gaze_angle[i, 0] = math.asin(-gaze[i, 1])
                        gaze_angle[i, 1] = math.atan2(-gaze[i, 0], -gaze[i, 2])
                        head_angle[i, 0] = math.asin(-head[i, 1])
                        head_angle[i, 1] = math.atan2(-head[i, 0], -head[i, 2])
                    in_x = input_data.reshape([self._batch_size, 36, 60, 1]) / 255.

                    sess_ldmks.run(train_op_ldmks, feed_dict={tf_x: in_x, tf_y_gaze: gaze_angle, tf_y_head: head_angle,
                                                              train_x: np.reshape(gaze[:, 0],
                                                                                  [self._batch_size, 1]),
                                                              train_y: np.reshape(-gaze[:, 1],
                                                                                  [self._batch_size, 1]),
                                                              train_z: np.reshape(gaze[:, 2],
                                                                                  [self._batch_size, 1])})

                    loss_ = sess_ldmks.run(loss_sub,
                                           feed_dict={tf_x: in_x, tf_y_gaze: gaze_angle, tf_y_head: head_angle})

                    pre = sess_ldmks.run(pre_gaze, feed_dict={tf_x: in_x, tf_y_gaze: gaze_angle, tf_y_head: head_angle})

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

                    subscript = np.random.randint(0, 213659 - 1, [self._batch_size])
                    right_img, right_gaze = test_.data_sources(subscript)
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

                        gaze = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_img, tf_y_head: test_pose})[0]
                        test_x = test_gaze[0]
                        test_y = test_gaze[1]
                        test_z = test_gaze[2]
                        x, y, z = two2three(gaze[0], gaze[1])

                        def cal_error(a_l, a_n, a_m, b_l, b_n, b_m):
                            # a = cal_angle(a_l, a_n, a_m)
                            # b = cal_angle(b_l, b_n, b_m)
                            res = math.acos((a_l * b_l + a_n * b_n + a_m * b_m) / (
                                    math.sqrt(a_l * a_l + a_n * a_n + a_m * a_m) * math.sqrt(
                                b_l * b_l + b_n * b_n + b_m * b_m))) / math.pi * 180
                            return abs(res)

                        error = cal_error(test_x, test_y, test_z, x, y, z)
                        ERROR += error
                    print(step, loss_, cal_acc(pre, gaze_angle), ERROR / (self._batch_size + 1))
                    if self._test_:
                        test_image_all(60, 36)
                    if self._is_save and step % 50 == 0:
                        print(Step, self._learn_rate, "-----------------  Have been save. ----------------")
                        saver_ldmks_mc.save(sess_ldmks, self._save_path)
                if Step % 5 == 0:
                    self._learn_rate *= 0.1

    """ ResNet* """

    def builds_ResNet_Direct_model(self):
        self._data_sources = BaseDateSource(60, 36, True)
        # self._count_data = 20188 // 2
        test_ = MPIIGaze.MPIIGAZE()
        self._is_save = True
        self._is_load = None
        self._save_path = './params/ResNet_Direct_UE'
        self._load_path = './params/ResNet_Direct_UE'
        self._batch_size = 256
        self._learn_rate = 0.001
        self._count_data = 102014 // 2

        ldmks_graph = tf.Graph()
        with ldmks_graph.as_default():

            tf_x = tf.placeholder(tf.float32, [None, 36, 60, 1], name="x_ldmks") / 1000.
            tf_y_gaze = tf.placeholder(tf.float32, [None, 2], name='y_angle')
            train_x = tf.placeholder(tf.float32, [None, 1], name='x_gaze')
            train_y = tf.placeholder(tf.float32, [None, 1], name='y_gaze')
            train_z = tf.placeholder(tf.float32, [None, 1], name='z_gaze')

            # network
            net = tf.layers.conv2d(tf_x, 24, 3, 1, 'same', activation=tf.nn.relu)  # [None, 36, 60, 1]
            # net = tf.layers.dropout(net, 0.3)
            net = tf.nn.relu(self.BN(net, 24))

            net_2 = self.block(net, 24, 2)

            net_3 = self.block(net_2, 24, 2)

            net_4 = self.block(net_3, 48, 2)

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

        def test_image_all(w, h):
            img1 = cv2.imread("./test/6660.jpg")
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.imread("./test/12.jpg")
            img2 = cv2.resize(img2, (w, h))
            img3 = cv2.imread("./test/999.jpg")
            img3 = cv2.resize(img3, (w, h))

            test_x1 = utils.pro_data(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
            test_x1 = test_x1.reshape(1, h, w, 1) / 255.
            pre_test1 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_x1})
            pre_gaze1 = pre_test1[0, 0:2]

            test_x2 = utils.pro_data(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)).reshape(1, h, w, 1) / 255.
            pre_test2 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_x2})
            pre_gaze2 = pre_test2[0, 0:2]

            test_x3 = utils.pro_data(cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)).reshape(1, h, w, 1) / 255.
            pre_test3 = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_x3})
            pre_gaze3 = pre_test3[0, 0:2]

            img_temp1 = img1.copy()
            img_temp2 = img2.copy()
            img_temp3 = img3.copy()

            cv2.arrowedLine(img_temp1, (128, 96), (int(pre_gaze1[0] * 40 + 128), int(pre_gaze1[1] * 40 + 96)),
                            (0, 255, 255), 2)
            cv2.imshow('eye1', img_temp1)

            cv2.arrowedLine(img_temp2, (128, 96), (int(pre_gaze2[0] * 40 + 128), int(pre_gaze2[1] * 40 + 96)),
                            (0, 255, 255), 2)
            cv2.imshow('eye2', img_temp2)

            cv2.arrowedLine(img_temp3, (128, 96), (int(pre_gaze3[0] * 40 + 128), int(pre_gaze3[1] * 40 + 96)),
                            (0, 255, 255), 2)
            cv2.imshow('eye3', img_temp3)

            # cv2.imshow('eyeb3', pre_mask_contour3)

            # print(step, ldmks_caruncle1[0][0], ldmks_caruncle2[0][0], ldmks_caruncle3[0][0])
            cv2.waitKey(1)

        with tf.Session(graph=ldmks_graph) as sess_ldmks:
            sess_ldmks.run(tf.global_variables_initializer())
            saver_ldmks_mc = tf.train.Saver()
            if self._is_load:
                saver_ldmks_mc.restore(sess_ldmks, self._load_path)

            gaze_angle = np.zeros((self._batch_size, 2))
            for Step in range(25):
                self._data_sources.init_num()

                for step in range(self._count_data // self._batch_size):

                    self._data_sources.record_data(self._batch_size)
                    input_data = np.array(self._data_sources.input_image())
                    gaze = np.array(self._data_sources.look_vec())
                    for i in range(self._batch_size):
                        gaze_angle[i, 0] = math.asin(-gaze[i, 1])
                        gaze_angle[i, 1] = math.atan2(-gaze[i, 0], -gaze[i, 2])
                    in_x = input_data.reshape([self._batch_size, 36, 60, 1]) / 255.

                    sess_ldmks.run(train_op_ldmks, feed_dict={tf_x: in_x, tf_y_gaze: gaze_angle,
                                                              train_x: np.reshape(gaze[:, 0],
                                                                                  [self._batch_size, 1]),
                                                              train_y: np.reshape(-gaze[:, 1],
                                                                                  [self._batch_size, 1]),
                                                              train_z: np.reshape(gaze[:, 2],
                                                                                  [self._batch_size, 1])})

                    loss_ = sess_ldmks.run(loss_sub, feed_dict={tf_x: in_x, tf_y_gaze: gaze_angle})

                    pre = sess_ldmks.run(pre_gaze, feed_dict={tf_x: in_x, tf_y_gaze: gaze_angle})

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

                    subscript = np.random.randint(0, 213659 - 1, [self._batch_size])
                    right_img, right_gaze = test_.data_sources(subscript)
                    ERROR = 0
                    for i in range(self._batch_size):
                        test_img = cv2.imread(right_img[i], 0)
                        test_img = np.array(cv2.resize(test_img, (60, 36)))
                        test_gaze = np.loadtxt(right_gaze[i])
                        test_img = utils.pro_data(test_img)
                        test_img = utils.Sharpen(test_img)
                        test_img = test_img.reshape([1, 36, 60, 1]) / 255.

                        gaze = sess_ldmks.run(pre_gaze, feed_dict={tf_x: test_img})[0]
                        test_x = test_gaze[0]
                        test_y = test_gaze[1]
                        test_z = test_gaze[2]
                        x, y, z = two2three(gaze[0], gaze[1])

                        def cal_error(a_l, a_n, a_m, b_l, b_n, b_m):
                            # a = cal_angle(a_l, a_n, a_m)
                            # b = cal_angle(b_l, b_n, b_m)
                            res = math.acos((a_l * b_l + a_n * b_n + a_m * b_m) / (
                                    math.sqrt(a_l * a_l + a_n * a_n + a_m * a_m) * math.sqrt(
                                b_l * b_l + b_n * b_n + b_m * b_m))) / math.pi * 180
                            return abs(res)

                        error = cal_error(test_x, test_y, test_z, x, y, z)
                        ERROR += error
                    print(step, loss_, cal_acc(pre, gaze_angle), ERROR / (self._batch_size + 1))
                    if self._test_:
                        test_image_all(60, 36)
                    if self._is_save and step % 50 == 0:
                        print(Step, self._learn_rate, "-----------------  Have been save. ----------------")
                        saver_ldmks_mc.save(sess_ldmks, self._save_path)
                if Step % 5 == 0:
                    self._learn_rate *= 0.1

    """ ResNet """

    def builds_ResNet_model(self):
        self._is_save = True
        self._is_load = None
        self._save_path = './params/ResNet_UE'
        self._load_path = './params/ResNet_UE'
        self._batch_size = 32
        self._learn_rate = 0.001
        self._data_sources = BaseDateSource(256, 192, True)

        # self._count_data = 45000
        ldmks_graph = tf.Graph()
        with ldmks_graph.as_default():

            tf_x = tf.placeholder(tf.float32, [None, 192, 256, 1], name="x_ldmks")
            tf_y = tf.placeholder(tf.float32, [None, 110], name='y_ldmks')

            # network
            net = tf.layers.conv2d(tf_x, 16, 3, 2, 'same', activation=tf.nn.relu)
            net = tf.layers.dropout(net, 0.3)
            net = tf.nn.relu(self.BN(net, 16))

            net_2 = self.block(net, 32, 2)

            net_3 = self.block(net_2, 64, 2)

            net_4 = self.block(net_3, 128, 2)

            net_5 = self.block(net_4, 256, 2)

            net_i = tf.reshape(net_5, [-1, 6 * 8 * 256])
            net_i = tf.layers.dense(net_i, 1000, tf.nn.relu)
            net_i = tf.layers.dropout(net_i, 0.5)
            pre_ldmks = tf.layers.dense(net_i, 110, name='pre_ldmks')

            def the_difference(pre, inputs):
                temp = tf.reduce_sum(tf.abs(tf.subtract(pre, inputs)))
                score = (temp + 1.) / (tf.abs(tf.reduce_sum(tf.add(pre, inputs))) + 1.)
                return score

            """ ldmks """
            loss_difference1 = the_difference(pre_ldmks, tf_y)
            loss_ldmks = 0.9 * loss_difference1
            train_op_ldmks = tf.train.AdamOptimizer(self._learn_rate).minimize(loss_ldmks)
            loss_ldmks = tf.cast(loss_ldmks, tf.float32, name='loss_ldmks_mc')
            pre_test_mc = tf.add(pre_ldmks, 0, name='pre_ldmks_mc')

            # self.stats_graph(ldmks_graph)

        with tf.Session(graph=ldmks_graph) as sess_ldmks:
            sess_ldmks.run(tf.global_variables_initializer())
            saver_ldmks_mc = tf.train.Saver()
            if self._is_load:
                saver_ldmks_mc.restore(sess_ldmks, self._load_path)

            for Step in range(25):
                self._data_sources.init_num()

                for step in range(self._count_data // self._batch_size):

                    self._data_sources.record_data(self._batch_size)
                    input_data = np.array(self._data_sources.input_image())
                    landmarks = np.array(self._data_sources.ldmks_data())
                    landmarks = landmarks.reshape((self._batch_size, 110))

                    in_x = input_data.reshape([self._batch_size, 192, 256, 1]) / 255.

                    loss, _ = sess_ldmks.run([loss_ldmks, train_op_ldmks], feed_dict={tf_x: in_x, tf_y: landmarks})

                    # acc = compute_accuracy(in_x, landmarks)
                    if step % 5 == 0:
                        print(Step, step, loss)
                if self._is_save:
                    print(Step, "------------- Have been save. --------------")
                    saver_ldmks_mc.save(sess_ldmks, self._save_path)
                if Step < 25 and Step % 5:
                    self._learn_rate *= 0.1

    def builds_KNN_model(self):
        KNN_model = KNeighborsRegressor(algorithm='auto', leaf_size=1, metric='minkowski', metric_params=None,
                                        n_jobs=None, n_neighbors=7, p=2, weights='distance')
        data_sources = BaseDateSource(15, 9, None)
        data_sources.init_num()
        data_sources.record_data(self._count_data)
        input_data = np.array(data_sources.input_image())
        gaze = np.array(data_sources.look_vec())
        gaze_angle = np.zeros((self._count_data, 2))
        for i in range(self._count_data):
            gaze_angle = np.zeros((self._count_data, 2))
            gaze_angle[i, 0] = math.asin(-gaze[i, 1])
            gaze_angle[i, 1] = math.atan2(-gaze[i, 0], -gaze[i, 2])
        in_x = input_data.reshape([-1, 15 * 9]) / 255.
        KNN_model.fit(in_x, gaze_angle)
        joblib.dump(KNN_model, "./params/KNN_UE.model")

    def builds_RF_model(self):
        RF_model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features=24,
                                         max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                         min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                                         n_estimators=150, n_jobs=None, oob_score=False, random_state=None,
                                         verbose=0, warm_start=False)
        print(RF_model)
        data_sources = BaseDateSource(15, 9, None)
        data_sources.init_num()
        data_sources.record_data(self._count_data)
        input_data = np.array(data_sources.input_image())
        gaze = np.array(data_sources.look_vec())
        gaze_angle = np.zeros((self._count_data, 2))
        for i in range(self._count_data):
            gaze_angle = np.zeros((self._count_data, 2))
            gaze_angle[i, 0] = math.asin(-gaze[i, 1])
            gaze_angle[i, 1] = math.atan2(-gaze[i, 0], -gaze[i, 2])
        in_x = input_data.reshape([-1, 15 * 9]) / 255.
        RF_model.fit(in_x, gaze_angle)
        joblib.dump(RF_model, "./params/RF_UE.model")

    def train_model(self, name):
        if name == 'KNN':
            self.builds_KNN_model()
        elif name == 'RF':
            self.builds_RF_model()
        elif name == 'GazeNet':
            self.builds_GazeNet_model()
        elif name == 'UEGazeNet':
            self.builds_UEGazeNet_Direct_model()
        elif name == 'UEGazeNet*':
            self.builds_UEGazeNet_model()
        elif name == 'UEGazeNet_gaze':
            self.builds_UEGazeNet_gaze_model()
        elif name == 'ResNet':
            self.builds_ResNet_Direct_model()
        elif name == 'ResNet*':
            self.builds_ResNet_model()


UE = ModelsUE()
UE.train_model('RF')
