from sklearn.externals import joblib
import tensorflow as tf
import numpy as np
import pykeyboard
import pymouse
import utils
import cv2
import os


mouse = pymouse.PyMouse()
key = pykeyboard.PyKeyboard()


class LANDMARKS(object):

    def __init__(self):
        self._load_ldmks_path = 'params/UEGazeNet'
        self._ldmks_graph = tf.Graph()
        with self._ldmks_graph.as_default():
            self._saver_ldmks = tf.train.import_meta_graph('params/UEGazeNet.meta')
        self._sess_ldmks = tf.Session(graph=self._ldmks_graph)
        self._saver_ldmks.restore(self._sess_ldmks, self._load_ldmks_path)
        self.prediction_ldmks = self._ldmks_graph.get_tensor_by_name('pre_ldmks_mc:0')
        self.tf_x = self._ldmks_graph.get_tensor_by_name('x_ldmks:0')

    def get_pre(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        test_x = np.array(img).reshape([1, 192, 256, 1]) / 255.
        
        test_pre = self._sess_ldmks.run(self.prediction_ldmks, feed_dict={self.tf_x: test_x})
        
        return test_pre


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
        gaze_pre = self._sess_gaze.run(self.prediction_gaze, feed_dict={self.tf_x: in_x / 256.})[0]
        x, y, z = utils.two2three(gaze_pre[0], gaze_pre[1])
        gaze_pre = [x, y]
        return gaze_pre


class GESTURE(object):

    def __init__(self):
        self._load_gaze_path = './params/gesture'
        self._gaze_graph = tf.Graph()
        with self._gaze_graph.as_default():
            self._saver_gaze = tf.train.import_meta_graph('./params/gesture.meta')
        self._sess_gaze = tf.Session(graph=self._gaze_graph)
        self._saver_gaze.restore(self._sess_gaze, self._load_gaze_path)
        self.prediction_gaze = self._gaze_graph.get_tensor_by_name('pre_gesture:0')
        self.tf_x = self._gaze_graph.get_tensor_by_name('tf_x:0')

    def get_pre(self, img):
        in_x = img
        pre = self._sess_gaze.run(self.prediction_gaze, feed_dict={self.tf_x: in_x})[0]

        return pre


class DRAWGESTURE(object):

    def __init__(self):
        self._left_top_l = np.zeros(2, dtype=np.float32)
        self._left_top_r = np.zeros(2, dtype=np.float32)
        self._right_top_l = np.zeros(2, dtype=np.float32)
        self._right_top_r = np.zeros(2, dtype=np.float32)
        self._left_bottom_l = np.zeros(2, dtype=np.float32)
        self._left_bottom_r = np.zeros(2, dtype=np.float32)
        self._right_bottom_l = np.zeros(2, dtype=np.float32)
        self._right_bottom_r = np.zeros(2, dtype=np.float32)
        self._la0 = 0
        self._la1 = 0
        self._ra0 = 0
        self._ra1 = 0
        self._Lvec10 = np.zeros(2, dtype=np.float32)
        self._Lvec01 = np.zeros(2, dtype=np.float32)
        self._Rvec10 = np.zeros(2, dtype=np.float32)
        self._Rvec01 = np.zeros(2, dtype=np.float32)

        self._screen_w, self._screen_h = utils.screen_size()
        self._Detect = 0
        self._write = 0
        self.LDMKS = LANDMARKS()
        self.GA = GAZE()
        self._root_path = './gesture_data/'  # 保存根目录
        self._pattern = 0
        # TODO 每次使用前需要检查的，在标签栏更改“用户编号”
        self._people = '0/'  # 15
        self._path = self._root_path + self._people + '%d/' % self._pattern  # 标签
        if os.path.exists(self._path) is False:
            os.makedirs(self._path)
        self._num = 0  # 起始编号

        self._blackboard_l = np.zeros((self._screen_h, self._screen_w, 1))
        self._blackboard_r = np.zeros((self._screen_h, self._screen_w, 1))
        self._x_l = int(0)
        self._y_l = int(0)
        self._x_r = int(0)
        self._y_r = int(0)
        self._point_L = []
        self._point_R = []
        self._gesture = GESTURE()

    def nothing(self, x):
        pass

    def work(self):
        capture_l = cv2.VideoCapture(0)
        capture_r = cv2.VideoCapture(1)
        ret, frame_l = capture_l.read()
        _, frame_r = capture_r.read()
        # cv2.namedWindow('eye_l', cv2.WINDOW_NORMAL)
        cv2.namedWindow('blackboard_l', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('blackboard_r', cv2.WINDOW_NORMAL)
        cv2.namedWindow('eye_r', cv2.WINDOW_NORMAL)

        while ret:
            frame_l = cv2.resize(frame_l, (256, 192))
            frame_l = cv2.flip(frame_l, -1)
            ldmks_l = self.LDMKS.get_pre(frame_l)
            look_vec_l = self.GA.get_gaze(ldmks_l)
            ldmks_interior_margin_l, ldmks_caruncle_l, ldmks_iris_l = utils.separate_all(1, ldmks_l)
            eye_c_l, eye_r_l = utils.get_r_c(ldmks_iris_l.reshape((32, 2)))
            look_vec_l = np.array(look_vec_l).reshape(2)
            utils.draw_eye_vec(frame_l, eye_c_l, list(look_vec_l))
            utils.draw_ldmks(frame_l, ldmks_interior_margin_l, ldmks_caruncle_l, ldmks_iris_l)

            frame_r = cv2.flip(cv2.resize(frame_r, (256, 192)), 1)
            ldmks_r = self.LDMKS.get_pre(frame_r)
            look_vec_r = self.GA.get_gaze(ldmks_r)
            ldmks_interior_margin_r, ldmks_caruncle_r, ldmks_iris_r = utils.separate_all(1, ldmks_r)
            eye_c_r, eye_r_r = utils.get_r_c(ldmks_iris_r.reshape((32, 2)))
            look_vec_r = np.array(look_vec_r).reshape(2)
            utils.draw_eye_vec(frame_r, eye_c_r, list(look_vec_r))
            utils.draw_ldmks(frame_r, ldmks_interior_margin_r, ldmks_caruncle_r, ldmks_iris_r)

            if self._Detect == 1:
                # coordinate_l = utils.get_gaze_point(self._left_top_l, self._right_bottom_l, look_vec_l[0], look_vec_l[1])
                # coordinate_r = utils.get_gaze_point(self._left_top_r, self._right_bottom_r, -look_vec_r[0], look_vec_r[1])
                # self._x_r, self._y_r = utils.normed(coordinate_r)
                self._x_l, self._y_l = utils.PM(self._la0, self._la1, look_vec_l[0], look_vec_l[1], self._Lvec10, self._Lvec01, self._left_top_l)
                self._x_r, self._y_r = utils.PM(self._ra0, self._ra1, look_vec_r[0], look_vec_r[1], self._Rvec10, self._Rvec01, self._left_top_r)
                coordinate_l = np.array((self._x_l, self._y_l))
                coordinate_r = np.array((self._x_r, self._y_r))
                self._x_l, self._y_l = utils.normed(coordinate_l, 'l')
                self._x_r, self._y_r = utils.normed(coordinate_r, 'r')
                if self._write == 0:
                    self._blackboard_l = np.zeros((self._screen_h, self._screen_w, 1))
                    self._blackboard_r = np.zeros((self._screen_h, self._screen_w, 1))
                    utils.drawTrajectory(self._blackboard_l, (self._x_l, self._y_l), eye='l')
                    utils.drawTrajectory(self._blackboard_r, (self._x_r, self._y_r), eye='r')
                elif self._write == 1:
                    utils.drawTrajectory(self._blackboard_l, (self._x_l, self._y_l), eye='l')
                    utils.drawTrajectory(self._blackboard_r, (self._x_r, self._y_r), eye='r')
                    self._point_L.append([self._x_l, self._y_l])
                    self._point_R.append([self._x_r, self._y_r])
                # print(self._blackboard_l.shape)
                # cv2.imshow('blackboard_r', self._blackboard_r)
                cv2.imshow('blackboard_l', np.concatenate([self._blackboard_l, self._blackboard_r], 0))

            cv2.imshow('L', cv2.imread(self._root_path + 'test/%d.jpg' % self._pattern))
            # cv2.imshow('eye_l', cv2.resize(frame_l, (512, 384)))
            cv2.imshow('eye_r', np.concatenate([cv2.resize(frame_l, (512, 384)), cv2.resize(cv2.flip(frame_r, 1), (512, 384))], 1))

            keycode = cv2.waitKey(1)
            if keycode == ord('w'):
                cv2.imwrite(self._path + '%s_L.jpg' % str(self._num), self._blackboard_l * 255)
                cv2.imwrite(self._path + '%s_R.jpg' % str(self._num), self._blackboard_r * 255)
                np.savetxt(self._path + '%s_L.txt' % str(self._num), np.array(self._point_L))
                np.savetxt(self._path + '%s_R.txt' % str(self._num), np.array(self._point_R))
                self._blackboard_l = np.zeros((self._screen_h, self._screen_w, 1))
                self._blackboard_r = np.zeros((self._screen_h, self._screen_w, 1))
                utils.init_the_before_d((self._x_l, self._y_l), 'l')
                utils.init_the_before_d((self._x_r, self._y_r), 'r')
                self._point_L = []
                self._point_R = []
                self._num += 1
                self._write = 0
            elif keycode == ord('e'):
                self._blackboard_l = np.zeros((self._screen_h, self._screen_w, 1))
                self._blackboard_r = np.zeros((self._screen_h, self._screen_w, 1))
                utils.init_the_before_d((self._x_l, self._y_l), 'l')
                utils.init_the_before_d((self._x_r, self._y_r), 'r')
                self._point_L = []
                self._point_R = []
                self._write = 0
            elif keycode == ord('q'):
                self._point_L = []
                self._point_R = []
                utils.init_the_before_d((self._x_l, self._y_l), 'l')
                utils.init_the_before_d((self._x_r, self._y_r), 'r')
                self._blackboard_l = np.zeros((self._screen_h, self._screen_w, 1))
                self._blackboard_r = np.zeros((self._screen_h, self._screen_w, 1))
                self._write = 1
                self._point_L.append([self._x_l, self._y_l])
                self._point_R.append([self._x_r, self._y_r])
            elif keycode == ord('a'):
                self._left_top_l[0] = look_vec_l[0]
                self._left_top_l[1] = look_vec_l[1]
                self._left_top_r[0] = look_vec_r[0]
                self._left_top_r[1] = look_vec_r[1]
                # print(self._left_top)
            elif keycode == ord('s'):
                self._right_top_l[0] = look_vec_l[0]
                self._right_top_l[1] = look_vec_l[1]
                self._right_top_r[0] = look_vec_r[0]
                self._right_top_r[1] = look_vec_r[1]
            elif keycode == ord('z'):
                self._left_bottom_l[0] = look_vec_l[0]
                self._left_bottom_l[1] = look_vec_l[1]
                self._left_bottom_r[0] = look_vec_r[0]
                self._left_bottom_r[1] = look_vec_r[1]
            elif keycode == ord('x'):
                self._right_bottom_l[0] = look_vec_l[0]
                self._right_bottom_l[1] = look_vec_l[1]
                self._right_bottom_r[0] = look_vec_r[0]
                self._right_bottom_r[1] = look_vec_r[1]
                # print(self._right_bottom)
            elif keycode == ord('c'):
                self._Detect = 1
                self._write = 0
                self._la0, self._la1, self._Lvec10, self._Lvec01 = utils.thePMvalue(self._left_top_l, self._right_top_l, self._left_bottom_l, self._right_bottom_l)
                self._ra0, self._ra1, self._Rvec10, self._Rvec01 = utils.thePMvalue(self._left_top_r, self._right_top_r, self._left_bottom_r, self._right_bottom_r)
                utils.init_the_before_d((self._x_l, self._y_l), 'l')
                # utils.init_the_before_d((self._x_r, self._y_r), 'r')
            elif keycode == ord('m'):
                b = 0
                a = np.reshape(cv2.resize(self._blackboard_r, (32, 32), interpolation=cv2.INTER_AREA), [1, 32 * 32])  # 1
                b = self._gesture.get_pre(a)
                a = np.reshape(cv2.resize(self._blackboard_l, (32, 32), interpolation=cv2.INTER_AREA), [1, 32 * 32])
                b = b + self._gesture.get_pre(a)
                print(b, np.where(b == b.max()))
                if cv2.waitKey(0) == 'w':
                    cv2.imwrite(self._path + '%s_L.jpg' % str(self._num), self._blackboard_l * 255)
                    cv2.imwrite(self._path + '%s_R.jpg' % str(self._num), self._blackboard_r * 255)
                    np.savetxt(self._path + '%s_L.txt' % str(self._num), np.array(self._point_L))
                    np.savetxt(self._path + '%s_R.txt' % str(self._num), np.array(self._point_R))
                self._blackboard_l = np.zeros((self._screen_h, self._screen_w, 1))
                self._blackboard_r = np.zeros((self._screen_h, self._screen_w, 1))
                utils.init_the_before_d((self._x_l, self._y_l), 'l')
                utils.init_the_before_d((self._x_r, self._y_r), 'r')
                self._point_L = []
                self._point_R = []
                self._write = 0
            elif keycode == ord('v'):
                self._Detect = 0
                self._write = 0
            elif keycode == ord('p'):
                break
            if self._num == 5:
                self._pattern += 1
                self._path = self._root_path + self._people + '%d/' % self._pattern  # 标签
                if os.path.exists(self._path) is False:
                    os.makedirs(self._path)
                self._num = 0  # 起始编号

            ret, frame_l = capture_l.read()
            _, frame_r = capture_r.read()


t = DRAWGESTURE()
t.work()
