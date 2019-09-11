# -*- coding: utf-8 -*-
# @Time    : 2019/5/9
# @Site    : Input the train data
# @File    : data_sources.py
# @Software: PyCharm

import cv2
import numpy as np
import json
from glob import glob
import utils
import math


class BaseDateSource(object):

    def __init__(self, size_w=256, size_h=192, data_enhancement=True):
        self._file_path = "./imgs_UnityEye/"
        self._json_fns = glob(self._file_path + "*.json")     # all json files
        self._num_all_data = int(0)
        self._input_image = []                          # the images of eye
        self._landmarks_data = []                           # the data of landmark
        self._landmarks_interior_margin = []                # interior margin`s landmark
        self._landmarks_caruncle = []                       # caruncle`s landmark
        self._landmarks_iris = []                           # iris`s landmark
        self._look_vec = []                             # gaze vectors
        self._head_vec = []                             # head pose
        self._eye_center = []                           # eye center
        self._eye_radius = []                           # eye radius
        self._size_w = size_w                           # the width of dataset you need
        self._size_h = size_h                           # the height of dataset you need
        self._isEnhancement = data_enhancement
        self._multiple_size = [1, 2, 4, 8, 16, 32]

        # self.record_data()

    def record_data(self, batch_size=100):
        self.del_data()
        start_num = self._num_all_data
        end_num = self._num_all_data + batch_size
        i = 0
        for json_fn in self._json_fns[start_num: end_num]:
            """ load the train data."""

            img = cv2.imread("/%s.jpg" % json_fn[52:-5], 0)
            data_file = open(json_fn)
            ldmks_data = json.load(data_file)

            """ load the landmarks and gaze. """

            def process_json_list(json_list):
                ldmks = [eval(s) for s in json_list]
                return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])

            def list_add(a, b):
                a = np.array(a)
                b = np.array(b)
                c = np.vstack((a, b))
                return c

            ldmks_interior_margin = process_json_list(ldmks_data['interior_margin_2d'])[:, :2]  # 16
            ldmks_caruncle = process_json_list(ldmks_data['caruncle_2d'])[:, :2]  # 7
            ldmks_iris = process_json_list(ldmks_data['iris_2d'])[:, :2]  # 32
            gaze = list(eval(ldmks_data['eye_details']['look_vec']))[:3]
            head_pose = np.array(eval(ldmks_data['head_pose']))[:3]  # 3
            head_pose = head_pose / 180. * math.pi
            M, _ = cv2.Rodrigues(head_pose)
            head = M[:, 2]

            gaze[1] = -gaze[1]

            """ edit the data. """
            """ resize. """
            ldmks_interior_margin[:, 0] = ldmks_interior_margin[:, 0] / 512. * self._size_w  # 16
            ldmks_caruncle[:, 0] = ldmks_caruncle[:, 0] / 512. * self._size_w  # 7
            ldmks_iris[:, 0] = ldmks_iris[:, 0] / 512. * self._size_w  # 32
            ldmks_interior_margin[:, 1] = ldmks_interior_margin[:, 1] / 384. * self._size_h  # 16
            ldmks_caruncle[:, 1] = ldmks_caruncle[:, 1] / 384. * self._size_h  # 7
            ldmks_iris[:, 1] = ldmks_iris[:, 1] / 384. * self._size_h  # 32

            """ Image transformation. """
            c, r = utils.get_r_c(ldmks_iris)
            temp = list_add(list_add(ldmks_interior_margin, ldmks_caruncle), ldmks_iris)

            # img = utils.blurred(img)
            img = cv2.resize(img, (self._size_w, self._size_h))
            if self._isEnhancement:
                if i % 3 == 0:
                    img, temp, gaze[:2] = utils.rotate_img(img, temp, c, gaze[:2])
                    img, temp = utils.zoom_img(img, temp, c)
                elif i % 3 == 1:
                    img, temp, gaze[:2] = utils.rotate_img(img, temp, c, gaze[:2])
                    img, temp = utils.move_img(img, temp, c)
                elif i % 3 == 2:
                    img, temp = utils.move_img(img, temp, c)
                    img, temp = utils.zoom_img(img, temp, c)
            i += 1

            """ production data set. """
            self._landmarks_interior_margin.append(ldmks_interior_margin)
            self._landmarks_caruncle.append(ldmks_caruncle)
            self._landmarks_iris.append(ldmks_iris)
            self._landmarks_data.append(temp)
            self._eye_center.append(c)
            self._eye_radius.append(r)
            self._input_image.append(img)
            self._look_vec.append(gaze)
            self._head_vec.append(head)

        self._num_all_data += next

    def del_data(self):
        self._input_image = []
        self._landmarks_interior_margin = []
        self._landmarks_caruncle = []
        self._landmarks_iris = []
        self._landmarks_data = []
        self._look_vec = []
        self._head_vec = []
        self._eye_radius = []
        self._eye_center = []

    def init_num(self):
        self._num_all_data = int(0)

    def num_all_data(self):
        return self._num_all_data

    def input_image(self):
        return self._input_image

    def ldmks_interior_margin(self):
        return self._landmarks_interior_margin

    def ldmks_caruncle(self):
        return self._landmarks_caruncle

    def ldmks_iris(self):
        return self._landmarks_iris

    def ldmks_data(self):
        return self._landmarks_data

    def look_vec(self):
        return np.array(self._look_vec)

    def head_vec(self):
        return np.array(self._head_vec)

    def json_fns(self):
        return self._json_fns

    def eye_center(self):
        return self._eye_center

    def eye_radius(self):
        return self._eye_radius

# input = BaseDateSource()
