import zipfile
import numpy as np
import pandas as pd
import cv2
import os
import math
from glob import glob
import shutil


class UT(object):
    def __init__(self):

        self._root_path = "./imgs_UT_Multviews/UT/data/"
        self._temp_peth = "./imgs_UT_Multviews/UT/temp/"
        self._list_dir = os.listdir(self._root_path)
        self._data_path = ''
        self._label_path = ''
        self._s_num = 0
        self._file_num = 0
        self._image_width = 60  # Pixel width and height.
        self._image_height = 36
        self._is_end = False
        self._is_zip = True
        self._rename_num = 0

    def read_dir(self):
        if self._s_num < len(self._list_dir):
            file_path = self._root_path + self._list_dir[self._s_num] + '/synth/'
            # print(self._list_dir[self._s_num])
            zip_list = glob(file_path + '*.zip')
            if self._file_num < len(zip_list):
                self._data_path = zip_list[self._file_num]
                # self._eye_direction = zip_list[]
                self._label_path = zip_list[self._file_num][:-4] + '.csv'
                self._file_num += 1
            else:
                self._file_num = 0
                self._s_num += 1
                self.read_dir()
        else:
            self._s_num = 0
            self._is_end = True
            print("------- all data reload -------")

    def test(self):
        while self._is_end is False:
            self.read_dir()
            data = zipfile.ZipFile(self._data_path)
            label = pd.read_csv(self._label_path, names=['gx', 'gy', 'gz', 'rx', 'ry', 'rz', 'tx', 'ty', 'tz', 'unknow'])
            label = label.drop(['rx', 'ry', 'rz', 'tx', 'ty', 'tz', 'unknow'], 1)
            label = np.array(label)
            dir_img = data.namelist()
            for img_file in dir_img:
                data.extract(img_file, self._temp_peth + "/tmp/")

            for img_num in range(len(dir_img)):
                test_img = cv2.imread(self._temp_peth + "/tmp/" + dir_img[img_num], 0)
                test_img = cv2.resize(test_img, (self._image_width, self._image_height))
                gaze = label[img_num, :]
                x = gaze[0]
                y = gaze[1]
                print(x, y)
                cv2.arrowedLine(test_img, (int(self._image_width / 2), int(self._image_height / 2)),
                                (int(x * 30 + self._image_width / 2), int(y * 30 + self._image_height / 2)), 255, 2)
                cv2.imshow("test", test_img)
                if cv2.waitKey(0) == ord('q'):
                    return 0
            data.close()

    def rename(self):
        while self._is_end is False:
            self.read_dir()
            data = zipfile.ZipFile(self._data_path)
            label = pd.read_csv(self._label_path, names=['gx', 'gy', 'gz', 'rx', 'ry', 'rz', 'tx', 'ty', 'tz', 'unknow'])
            label = label.drop(['tx', 'ty', 'tz', 'unknow'], 1)
            label = np.array(label)
            dir_img = data.namelist()
            for img_file in dir_img:
                data.extract(img_file, self._temp_peth + "/tmp")

            for img_num in range(len(dir_img)):
                test_img = cv2.imread(self._temp_peth + "/tmp/" + dir_img[img_num])
                # cv2.imshow("sss", test_img)
                # cv2.waitKey(0)
                gaze = label[img_num, :]
                cv2.imwrite(self._temp_peth+"%d.jpg" % self._rename_num, test_img)
                np.savetxt(self._temp_peth+"%d.txt" % self._rename_num, gaze)
                if self._rename_num % 10000 == 0:
                    print(self._rename_num, self._s_num)
                self._rename_num += 1
            data.close()

    def make_train_data(self):
        subscript = np.random.randint(0, 2304144-1, [64000])
        data = np.ones((1, 60*36))
        label = np.ones((1, 3))
        for i in subscript:
            img = cv2.imread(self._temp_peth + "%d.jpg" % i)
            gaze = np.loadtxt(self._temp_peth + "%d.txt" % i, dtype=np.float32)
            img = np.reshape(img, [1, 60*36])
            gaze = np.reshape(gaze, [1, 3])
            data = np.append(data, img, 0)
            label = np.append(label, gaze, 0)
        data = np.delete(data, 0, 0)
        label = np.delete(label, 0, 0)
        np.savetxt(self._temp_peth + "/train_data/train_data_1.txt", data, fmt='%d')
        np.savetxt(self._temp_peth + "/train_data/train_label_1.txt", label)
        print(label.shape)
        # np.savetxt(self._temp_peth + "%d.txt" % self._rename_num, gaze)

    def get_train_data(self):
        img = np.loadtxt(self._temp_peth + "/train_data/train_data.txt", dtype=np.float32)
        gaze = np.loadtxt(self._temp_peth + "/train_data/train_label.txt", dtype=np.float32)
        # print(gaze[:3, :])
        return img, gaze

    def data_sources(self, subscript, step, batch, w=60, h=36):

        data = []
        gtwo = []
        gthree = []
        htwo = []
        hthree = []
        for i in subscript[step*batch:(step+1)*batch]:
            img = cv2.imread(self._temp_peth + "%d.jpg" % i, 0)
            img = cv2.resize(img, (h, w))
            gaze = np.loadtxt(self._temp_peth + "%d.txt" % i, dtype=np.float32)
            gx = gaze[0]
            gy = gaze[1]
            gz = gaze[2]

            # print(hx, hy, hz)
            M, _ = cv2.Rodrigues(gaze[3:])
            # print(M[:, 2])
            hx = M[0, 2]
            hy = M[1, 2]
            hz = M[2, 2]

            data.append(img)
            gtwo.append([math.asin(-gy), math.atan2(-gx, -gz)])
            gthree.append([gx, gy, gz])
            htwo.append([math.asin(-hy), math.atan2(-hx, -hz)])
            hthree.append([hx, hy, hz])
        data = np.array(data)
        g2 = np.array(gtwo)
        g3 = np.array(gthree)
        h2 = np.array(htwo)
        h3 = np.array(hthree)
        # print(two)
        # print("--------------------------",data.shape, label.shape,"----------------------------------")
        return data, g2, g3, h2, h3


# ut = UT()
# subscript = np.random.randint(0, 2304144 - 1, [64000])
# ut.data_sources(subscript, 0, 32)
# ut.make_train_data()
# ut.rename()
# ut.get_train_data()
# ut.test()
