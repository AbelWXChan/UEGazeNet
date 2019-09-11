import numpy as np
import os
import scipy.io as sio
import cv2
import utils
import math
import tensorflow as tf
from sklearn.externals import joblib
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

    def reader(self):
        reader = tf.train.NewCheckpointReader(self._load_ldmks_path)

        variables = reader.get_variable_to_shape_map()
        ops = self._ldmks_graph.get_operations()
        # var = tf.get_collection()

        for ele in variables:
            print(ele)

    def get_pre(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        test_x = np.array(img).reshape(1, 36, 60, 1) / 255.
        test_pre = self._sess_ldmks.run(self.prediction_ldmks, feed_dict={self.tf_x: test_x})
        # print(test_pre.shape)
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
        gaze_pre = self._sess_gaze.run(self.prediction_gaze, feed_dict={self.tf_x: in_x / 60.})
        return gaze_pre


class GAZE_direct(object):

    def __init__(self):
        self._load_gaze_path = './params/ResNet_Direct_UE'
        self._gaze_graph = tf.Graph()
        with self._gaze_graph.as_default():
            self._saver_gaze = tf.train.import_meta_graph('params/ResNet_Direct_UE.meta')
        self._sess_gaze = tf.Session(graph=self._gaze_graph)
        self._saver_gaze.restore(self._sess_gaze, self._load_gaze_path)
        self.prediction_gaze = self._gaze_graph.get_tensor_by_name('pre_gaze:0')
        self.tf_x = self._gaze_graph.get_tensor_by_name('x_ldmks:0')

    def reader(self):
        reader = tf.train.NewCheckpointReader(self._load_gaze_path)

        variables = reader.get_variable_to_shape_map()
        ops = self._gaze_graph.get_operations()
        # var = tf.get_collection()

        for ele in variables:
            print(ele)

    def get_pre(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        test_x = np.array(img).reshape(1, 36, 60, 1) / 255.
       
        test_pre = self._sess_gaze.run(self.prediction_gaze, feed_dict={self.tf_x: test_x})
        return test_pre


class MPIIGAZE(object):
    def __init__(self):

        self._root_path = './imgs_MPIIGaze/Data/Normalized/'
        self._temp_path = './imgs_MPIIGaze/Data/temp/'
        self._list_dir = os.listdir(self._root_path)
        self._file_path = ''
        self._people_num = 0
        self._day_num = 0
        self._image_width = 60
        self._image_height = 36
        self._ldmks = LANDMARKS()
        self._gaze = GAZE()

        self._is_end = None
        self._rename_num = 0
        self.all_num = 213659

    def read_dir(self):
        if self._people_num < len(self._list_dir):
            file_list = os.listdir(self._root_path + self._list_dir[self._people_num])
            if self._day_num < len(file_list):
                self._file_path = self._root_path + self._list_dir[self._people_num] + "/" + file_list[self._day_num]
                self._day_num += 1
            else:
                self._day_num = 0
                self._people_num += 1
                self.read_dir()
        else:
            self._people_num = 0
            # self._is_end = True
            print("------- all data reload -------")

    def get_dir(self):
        return self._file_path

    def read_matfile(self):
        self.read_dir()
        mat_contents = sio.loadmat(self._file_path)
        data = mat_contents['data']
        right = data['right']
        left = data['left']
        tmp = left[0, 0]
        _gaze = tmp['gaze'][0, 0]
        _img = tmp['image'][0, 0]
        _pose = tmp['pose'][0, 0]
        return _gaze, _img, _pose

    def rename(self):
        label = np.zeros([6])
        while self._is_end is None:
            right_gaze, right_img, right_pose = self.read_matfile()
            # print(right_pose)
            len_img, _ = right_gaze.shape
            # print(len_img)
            for i in range(len_img):
                img = np.array(cv2.resize(right_img[i, :, :], (60, 36)))
                label[0] = right_gaze[i, 0]
                label[1] = right_gaze[i, 1]
                label[2] = right_gaze[i, 2]
                label[3] = right_pose[i, 0]
                label[4] = right_pose[i, 1]
                label[5] = right_pose[i, 2]
                cv2.imwrite(self._temp_path + "%d.jpg" % self._rename_num, img)
                np.savetxt(self._temp_path + "%d.txt" % self._rename_num, label)
                self._rename_num += 1
            print(self._rename_num, self._day_num)  # 213658
            if self._rename_num > 213658:
                break

    def data_sources(self, subscript):

        data = []
        label = []
        for i in subscript:
            img = self._temp_path + "%d.jpg" % i
            gh = self._temp_path + "%d.txt" % i

            data.append(img)
            label.append(gh)
        data = np.array(data)
        label = np.array(label)
        # print(data[0],label[0])
        return data, label

    def test(self):
        test_num = 0
        right_gaze, right_img, right_pose = self.read_matfile()
        while True:
            test_img = np.array(cv2.resize(right_img[test_num, :, :], (self._image_width, self._image_height)))
            test_gaze = right_gaze[test_num, :]
            x = test_gaze[0]
            y = test_gaze[1]
            print(x, y, test_img.shape)
            cv2.arrowedLine(test_img, (int(self._image_width / 2), int(self._image_height / 2)),
                            (int(x * 100 + self._image_width / 2), int(y * 100 + self._image_height / 2)), 255, 1)
            cv2.imshow("test", test_img)
            if cv2.waitKey(0) == ord('q'):
                break
            test_num += 1

    def work_ldmks(self):
        self._image_width = 60
        self._image_height = 36
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)

        def cal_error(a_l, a_n, a_m, b_l, b_n, b_m):
            # a = cal_angle(a_l, a_n, a_m)
            # b = cal_angle(b_l, b_n, b_m)
            res = math.acos((a_l*b_l + a_n*b_n + a_m*b_m) / (math.sqrt(a_l*a_l + a_n*a_n + a_m*a_m)*math.sqrt(b_l*b_l + b_n*b_n + b_m*b_m))) / math.pi * 180
            return abs(res)

        test_num = 0
        ERROR = 0.
        subscript = np.random.randint(0, self.all_num - 1, [45000])
        right_img, right_gaze = self.data_sources(subscript)
        # print(len_img)
        for i in range(45000):
            test_img = cv2.imread(right_img[i], 0)
            test_img = np.array(cv2.resize(test_img, (60, 36)))
            test_gaze = np.loadtxt(right_gaze[i])

            ldmks = self._ldmks.get_pre(test_img)
            gaze = self._gaze.get_gaze(ldmks)
            test_x = test_gaze[0]
            test_y = test_gaze[1]
            test_z = test_gaze[2]
            theta = gaze[0][0]
            phi = gaze[0][1]
            x, y, z = utils.two2three(theta, phi)
            # test_theta = math.asin(-test_y)
            # test_phi = math.atan2(-test_x, -test_z)
            # theta = math.asin(-y)
            # phi = math.atan2(-x, -z)
            error = cal_error(test_x, test_y, test_z, x, y, z)
            ERROR += error
            if test_num % 100 == 0:
                print(ERROR / (test_num + 1))
            ldmks_interior_margin1, ldmks_caruncle1, ldmks_iris1 = utils.separate_all(1, ldmks)
            utils.draw_ldmks(test_img, ldmks_interior_margin1, ldmks_caruncle1, ldmks_iris1)
            cv2.arrowedLine(test_img, (int(self._image_width / 2), int(self._image_height / 2)),
                            (int(x * 100 + self._image_width / 2), int(y * 100 + self._image_height / 2)), 255, 2)
            cv2.arrowedLine(test_img, (int(self._image_width / 2), int(self._image_height / 2)),
                            (int(test_x * 100 + self._image_width / 2), int(test_y * 100 + self._image_height / 2)),
                            255, 1)
            cv2.imshow("test", test_img)
            cv2.waitKey(0)
            test_num += 1

    def work_ml_knn_rf(self):
        def cal_error(a_l, a_n, a_m, b_l, b_n, b_m):
            # a = cal_angle(a_l, a_n, a_m)
            # b = cal_angle(b_l, b_n, b_m)
            res = math.acos((a_l*b_l + a_n*b_n + a_m*b_m) / (math.sqrt(a_l*a_l + a_n*a_n + a_m*a_m)*math.sqrt(b_l*b_l + b_n*b_n + b_m*b_m))) / math.pi * 180
            return abs(res)

        test_num = 0
        ERROR = 0.
        error_list = []
        while self._is_end is None:
            subscript = np.random.randint(0, self.all_num - 1, [45000])
            right_img, right_gaze = self.data_sources(subscript)
            # print(len_img)
            model = joblib.load("./params/RF_UE.model")
            for i in range(45000):
                test_img = cv2.imread(right_img[i], 0)
                test_img = np.array(cv2.resize(test_img, (15, 9)))
                test_img = test_img.reshape([1, 15*9])
                test_gaze = np.loadtxt(right_gaze[i])
                gaze = model.predict(test_img)
                test_x = test_gaze[0]
                test_y = test_gaze[1]
                test_z = test_gaze[2]
                theta = gaze[0][0]
                phi = gaze[0][1]
                x, y, z = utils.two2three(theta, phi)
                error = cal_error(test_x, test_y, test_z, x, y, z)
                ERROR += error
                if test_num % 100 == 0:
                    # error_list.append(ERROR / (test_num + 1))
                    print(i, ERROR / (test_num + 1))
                    # ERROR = 0.
                    # test_num = 0
                test_num += 1
            # ET = pd.DataFrame(np.array(error_list))
            # ET.to_csv("./error_data/RF_MP.txt")

                # cv2.arrowedLine(test_img, (int(self._image_width / 2), int(self._image_height / 2)),
                #                 (int(x * 100 + self._image_width / 2), int(y * 100 + self._image_height / 2)), 255, 2)
                # cv2.arrowedLine(test_img, (int(self._image_width / 2), int(self._image_height / 2)),
                #                (int(test_x * 100 + self._image_width / 2), int(test_y * 100 + self._image_height / 2)),
                #                 255, 1)
                # cv2.imshow("test", test_img)
                # cv2.waitKey(0)

    def work_gaze(self):

        self._gaze_d = GAZE_direct()

        def cal_error(a_l, a_n, a_m, b_l, b_n, b_m):
            # a = cal_angle(a_l, a_n, a_m)
            # b = cal_angle(b_l, b_n, b_m)
            res = math.acos((a_l * b_l + a_n * b_n + a_m * b_m) / (math.sqrt(a_l * a_l + a_n * a_n + a_m * a_m) * math.sqrt(
                b_l * b_l + b_n * b_n + b_m * b_m))) / math.pi * 180
            return abs(res)

        def two2three(n, m):
            x = -math.cos(n) * math.sin(m)
            y = -math.sin(n)
            z = -math.cos(n) * math.cos(m)
            return x, y, z

        test_num = 0
        ERROR = 0.
        subscript = np.random.randint(0, self.all_num - 1, [45000])
        right_img, right_gaze = self.data_sources(subscript)
        # print(len_img)
        for i in range(45000):
            test_img = cv2.imread(right_img[i], 0)
            test_img = np.array(cv2.resize(test_img, (60, 36)))
            test_gaze = np.loadtxt(right_gaze[i])

            gaze = self._gaze_d.get_pre(test_img)[0]
            test_x = test_gaze[0]
            test_y = test_gaze[1]
            test_z = test_gaze[2]
            x, y, z = two2three(gaze[0], gaze[1])
            # x = gaze[0]
            # y = gaze[1]
            # z = gaze[2]
            x = 0. # np.random.randint(-2000, 2000, 1)/10000.
            y = 0. # np.random.randint(100, 3000, 1)/10000.
            z = -1. # np.random.randint(-1000, -950, 1)/1000.
            # test_img = np.array(cv2.resize(right_img[i, :, :], (self._image_width, self._image_height)))
            # test_theta = math.asin(-test_y)
            # test_phi = math.atan2(-test_x, -test_z)
            # theta = math.asin(-y)
            # phi = math.atan2(-x, -z)
            error = cal_error(test_x, test_y, test_z, x, y, z)
            ERROR += error
            if test_num % 100 == 0:
                print(ERROR / (test_num + 1))
                print("real: ", test_x, test_y, test_z, "\npre: ", x, y, z)
            # cv2.arrowedLine(test_img, (int(self._image_width / 2), int(self._image_height / 2)),
            #                 (int(x * 100 + self._image_width / 2), int(y * 100 + self._image_height / 2)), 255, 2)
            # cv2.arrowedLine(test_img, (int(self._image_width / 2), int(self._image_height / 2)),
            #                 (int(test_x * 100 + self._image_width / 2), int(test_y * 100 + self._image_height / 2)),
            #                 255, 1)
            # cv2.imshow("test", test_img)
            # cv2.waitKey(0)
            test_num += 1

    def statistics(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        test_num = 0
        gaze = []
        subscript = np.random.randint(0, self.all_num - 1, [45000])
        right_img, right_gaze = self.data_sources(subscript)
        for i in range(45000):
            test_gaze = np.loadtxt(right_gaze[i])
            # print(test_gaze.shape)
            gaze.append(test_gaze)
            if test_num % 1000 == 0:
                # f, axes = plt.subplots(2, 2, figsize=(7, 7))
                x = np.array(gaze)[:, 0]
                # x = x.reshape([-1, 1])
                # print(x)
                y = np.array(gaze)[:, 1]
                z = np.array(gaze)[:, 2]
                sns.set(palette="muted", color_codes=True)
                sns.distplot(x, kde=False, rug=True)
                sns.distplot(y, kde=False, rug=True)
                sns.distplot(z, kde=False, rug=True)
                plt.show()
            test_num += 1



