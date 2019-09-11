# -*- coding: utf-8 -*-
# @Time    : 2019/5/9
# @Site    : 一些函数与部分常量
# @File    : utils.py
# @Software: PyCharm

import numpy as np
import random
# import pyautogui
import math
import queue
import json
import cv2

BATCH_SIZE_GAZE = 64
LR_GAZE = 0.001
BATCH_SIZE_LDMKS = 32
LR_LDMKS = 0.001
count_data = 50000  # 44059
normed_point_l = queue.Queue()
normed_point_r = queue.Queue()
before_n_l = np.zeros(2, dtype=np.float32)
before_n_r = np.zeros(2, dtype=np.float32)
img_switch = cv2.imread('switch.jpg', 0)


def screen_size():
    width = 1
    height = 1
    # width, height = pyautogui.size()
    print(width, height)
    return width, height


screen_w, screen_h = screen_size()


def get_gaze_point(lt, rb, x, y):  # input: point
    Coordinate = np.zeros(2, dtype=np.float32)
    x = -x
    if x < lt[0]:
        x = lt[0]
    elif x > rb[0]:
        x = rb[0]
    if y > lt[1]:
        y = lt[1]
    elif y < rb[1]:
        y = rb[1]

    w = rb[0] - lt[0]
    h = lt[1] - rb[1]
    look_point_w = x - lt[0]
    look_point_h = lt[1] - y

    Coordinate[1] = screen_h * (look_point_h + 1e-10) / h
    Coordinate[0] = screen_w * (look_point_w + 1e-10) / w

    return Coordinate


def Sharpen(img):
    kernel = np.array(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]]
    )
    dst = cv2.filter2D(img, -1, kernel)
    return dst


def ContrastAndBrightness(img, con, bri):
    temp = np.uint8(np.clip(con * img + bri, 0, 255))
    return temp


def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    # print len(channels)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def equalizeHist(img):
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7, 7))
    # cl1 = clahe.apply(img)
    return cv2.equalizeHist(img)


def load_a_json(name, size_w, size_h):
    json_fns = "%s.json" % name
    data_file = open(json_fns)
    data = json.load(data_file)

    def process_json_list(json_list):
        ldmks = [eval(s) for s in json_list]
        return np.array([(x, 384 - y, z) for (x, y, z) in ldmks])

    def list_add(a, b):
        a = np.array(a)
        b = np.array(b)
        c = np.vstack((a, b))
        return c

    ldmks_interior_margin = process_json_list(data['interior_margin_2d'])[:, :2]
    ldmks_caruncle = process_json_list(data['caruncle_2d'])[:, :2]
    ldmks_iris = process_json_list(data['iris_2d'])[:, :2]

    ldmks_interior_margin[:, 0] = ldmks_interior_margin[:, 0] / 512. * size_w  # 16
    ldmks_caruncle[:, 0] = ldmks_caruncle[:, 0] / 512. * size_w  # 7
    ldmks_iris[:, 0] = ldmks_iris[:, 0] / 512. * size_w  # 32
    ldmks_interior_margin[:, 1] = ldmks_interior_margin[:, 1] / 384. * size_h  # 16
    ldmks_caruncle[:, 1] = ldmks_caruncle[:, 1] / 384. * size_h  # 7
    ldmks_iris[:, 1] = ldmks_iris[:, 1] / 384. * size_h  # 32

    temp = list_add(list_add(ldmks_interior_margin, ldmks_caruncle), ldmks_iris)

    return temp


def separate_all(BATCH_SIZE, ldmks):  # 110
    ldmks_interior_margin = []
    ldmks_caruncle = []
    ldmks_iris = []
    ldmks = np.array(ldmks)
    # print(ldmks.shape)
    ldmks = ldmks.reshape(55 * BATCH_SIZE, 2)
    for i in range(BATCH_SIZE):
        ldmks_interior_margin.append(ldmks[i * BATCH_SIZE:i * BATCH_SIZE + 16])
        ldmks_caruncle.append(ldmks[i * BATCH_SIZE + 16:i * BATCH_SIZE + 23])
        ldmks_iris.append(ldmks[i * BATCH_SIZE + 23:i * BATCH_SIZE + 55])
    return np.array(ldmks_interior_margin), np.array(ldmks_caruncle), np.array(ldmks_iris)  # 7, 16, 32


def separate_contour_and_iris(BATCH_SIZE, ldmks):  # 110
    ldmks_im_c = []
    ldmks_iris = []
    ldmks = np.array(ldmks)
    ldmks = ldmks.reshape(55 * BATCH_SIZE, 2)
    for i in range(BATCH_SIZE):
        ldmks_im_c.append(ldmks[i * BATCH_SIZE:i * BATCH_SIZE + 23])
        ldmks_iris.append(ldmks[i * BATCH_SIZE + 23:i * BATCH_SIZE + 55])
    # print(np.array(ldmks_im_c).shape)
    return np.array(ldmks_im_c), np.array(ldmks_iris)  # 23, 32


def ldmks_point_one(BATCH_SIZE, ldmks, name=None):
    ldmks = np.array(ldmks)
    if name == "1":
        ldmks = ldmks.reshape(BATCH_SIZE, 16, 2)
    elif name == "2":
        ldmks = ldmks.reshape(BATCH_SIZE, 7, 2)
    elif name == "3":
        ldmks = ldmks.reshape(BATCH_SIZE, 32, 2)
    return ldmks


def ldmks_point_im_c(ldmks):  # 46
    ldmks = np.array(ldmks)
    ldmks = ldmks.reshape(23, 2)
    return ldmks[:16], ldmks[16:23]  # 16, 7


def get_r_c(ldmks_iris):
    eye_c = np.mean(ldmks_iris[:, :2], axis=0).astype(int)
    r = np.mean(((ldmks_iris[:, 0] - eye_c[0]) ** 2 + (ldmks_iris[:, 1] - eye_c[1]) ** 2) ** 0.5)
    return eye_c, r


def draw_ldmks(img, ldmks_interior_margin, ldmks_caruncle, ldmks_iris):
    ldmks_interior_margin = ldmks_interior_margin.reshape((16, 2))
    ldmks_caruncle = ldmks_caruncle.reshape((7, 2))
    ldmks_iris = ldmks_iris.reshape((32, 2))
    for ldmk in np.vstack([ldmks_interior_margin, ldmks_caruncle, ldmks_iris[::2]]):
        cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 3, (0, 0, 0), -1)
    cv2.polylines(img, np.array([ldmks_interior_margin[:, :2]], int), True, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.polylines(img, np.array([ldmks_iris[:, :2]], int), True, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw green foreground points and lines
    for ldmk in np.vstack([ldmks_interior_margin, ldmks_caruncle, ldmks_iris[::2]]):
        cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 2, (0, 255, 0), -1)
    cv2.polylines(img, np.array([ldmks_interior_margin[:, :2]], int), True, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.polylines(img, np.array([ldmks_iris[:, :2]], int), True, (0, 255, 0), 1, cv2.LINE_AA)


def draw_ldmks_interior_margin(img, ldmks_interior_margin):
    for ldmk in ldmks_interior_margin[0]:
        cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 3, (0, 0, 0), -1)
    cv2.polylines(img, np.array([ldmks_interior_margin[0][:, :2]], int), True, (0, 0, 0), 2)

    for ldmk in ldmks_interior_margin[0]:
        cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 2, (0, 255, 0), -1)
    cv2.polylines(img, np.array([ldmks_interior_margin[0][:, :2]], int), True, (0, 255, 0), 1)


def draw_ldmks_caruncle(img, ldmks_caruncle):
    for ldmk in ldmks_caruncle[0]:
        cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 3, (0, 0, 0), -1)

    for ldmk in ldmks_caruncle[0]:
        cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 2, (0, 255, 0), -1)


def draw_ldmks_iris(img, ldmks_iris):
    for ldmk in np.vstack([ldmks_iris[0][::2]]):
        cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 3, (0, 0, 0), -1)
    cv2.polylines(img, np.array([ldmks_iris[0][:, :2]], int), True, (0, 0, 0), 2)

    for ldmk in np.vstack([ldmks_iris[0][::2]]):
        cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 2, (0, 255, 0), -1)
    cv2.polylines(img, np.array([ldmks_iris[0][:, :2]], int), True, (0, 255, 0), 1)


def draw_eye_vec(img, eye_c, look_vec):
    # look_mod = (look_vec[1]**2+look_vec[0]**2)**0.5
    # look_vec /= look_mod
    cv2.line(img, (eye_c[0], eye_c[1]), (int(eye_c[0] + look_vec[0] * 160), int(eye_c[1] + look_vec[1] * 160)), (0, 0, 0),
             3)
    cv2.arrowedLine(img, (eye_c[0], eye_c[1]), (int(eye_c[0] + look_vec[0] * 160), int(eye_c[1] + look_vec[1] * 160)),
             (0, 255, 255), 2)


def get_mask(point, size):
    mask_iris = np.zeros(size, np.uint8)
    mask_contour = np.zeros(size, np.uint8)

    ldmks_interior_margin, _, ldmks_iris = separate_all(1, point)
    ldmks_interior_margin = ldmks_interior_margin.reshape(16, 2)
    ldmks_iris = ldmks_iris.reshape(32, 2)
    # print(ldmks_interior_margin)
    mask_iris = cv2.fillConvexPoly(mask_iris, ldmks_iris[:, :2].astype(np.int32), 255)
    mask_contour = cv2.fillConvexPoly(mask_contour, ldmks_interior_margin[:, :2].astype(np.int32), 255)
    mask_iris = mask_iris + mask_contour - 255
    mask_iris = np.maximum(mask_iris, 0)

    return mask_contour // 255, mask_iris // 255


# smooth the movement of gaze
def normed(point, eye='l'):
    global before_n_l, before_n_r
    if eye == 'l':
        before_n_l[0] += point[0]
        before_n_l[1] += point[1]
        normed_point_l.put(point)
        size_point = normed_point_l.qsize()
        if size_point > 20:
            del_point = normed_point_l.get()
            before_n_l[0] -= del_point[0]
            before_n_l[1] -= del_point[1]
        x = before_n_l[0] / size_point
        y = before_n_l[1] / size_point
        # print(size_point, point, [x, y])
        return int(x), int(y)
    elif eye == 'r':
        before_n_r[0] += point[0]
        before_n_r[1] += point[1]
        normed_point_r.put(point)
        size_point = normed_point_r.qsize()
        if size_point > 20:
            del_point = normed_point_r.get()
            before_n_r[0] -= del_point[0]
            before_n_r[1] -= del_point[1]
        x = before_n_r[0] / size_point
        y = before_n_r[1] / size_point
        # print(size_point, point, [x, y])
        return int(x), int(y)


def init_the_before_d(point, eye='l'):
    global before_d_l, before_d_r
    if eye == 'l':
        before_d_l = [point, point]
        print(before_d_l)
    elif eye == 'r':
        before_d_r = [point, point]
        print(before_d_r)


def drawTrajectory(blackboard, point, eye='l'):
    global before_d_l, before_d_r
    if eye == 'l':
        before_d_l = list(before_d_l)
        if len(before_d_l) <= 10:
            before_d_l.append(point)
        else:
            before_d_l.pop(0)
            before_d_l.append(point)
        pts = np.array(before_d_l[:], int).reshape((-1, 1, 2))
        # print(pts, blackboard.shape)
        cv2.polylines(blackboard, [pts], False, 1, 20)
    elif eye == 'r':
        before_d_r = list(before_d_r)
        if len(before_d_r) <= 10:
            before_d_r.append(point)
        else:
            before_d_r.pop(0)
            before_d_r.append(point)
        pts = np.array(before_d_r[:], int).reshape((-1, 1, 2))
        # print(pts, blackboard.shape)
        cv2.polylines(blackboard, [pts], False, 1, 20)


def draw_acc_points(img, point):
    cv2.circle(img, (int(point[0]), int(point[1])), 15, (0, 0, 0), -1)
    cv2.circle(img, (int(point[0]), int(point[1])), 14, (0, 255, 0), -1)


def make_mask(img1, img2):
    img1 = img1.reshape(192, 256)
    img1 = - 255 * (np.round(img1).astype(np.uint8) - 1)
    img2 = img2.reshape(192, 256)
    img2 = - 255 * (np.round(img2).astype(np.uint8) - 1)
    mask = 255 * np.ones([192, 256, 3])
    mask[:, :, 1] = img1
    mask[:, :, 2] = img2

    return mask


def pro_data(img1):
    img1 = equalizeHist(img1)
    # img1 = cv2.medianBlur(img1, 13)
    # img1 = Sharpen(img1)
    # img1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 3)
    return img1


def move_img(img, point, c):  # done
    rows, cols = img.shape[:2]
    dst_point = point
    a = random.randint(-c[0] // 2, (cols - c[0]) // 2)
    b = random.randint(-c[1] // 2, (rows - c[1]) // 2)
    M = np.float32([[1, 0, a], [0, 1, b]])
    dst_img = cv2.warpAffine(img, M, (cols, rows))
    dst_point[:, 0] = point[:, 0] + a
    dst_point[:, 1] = point[:, 1] + b
    return dst_img, dst_point


def rotate_img(img, point, c, gaze):  # done
    rows, cols = img.shape[:2]
    dst_point = point
    dst_gaze = gaze
    dst_point[:, 0] = dst_point[:, 0] - c[0]
    dst_point[:, 1] = dst_point[:, 1] - c[1]
    angle = random.randint(-30, 30)
    r, theta = cv2.cartToPolar(gaze[0], gaze[1], angleInDegrees=True)
    x, y = cv2.polarToCart(r[0], theta[0] - angle, angleInDegrees=True)
    dst_gaze[0] = x[0, 0]
    dst_gaze[1] = y[0, 0]
    M = cv2.getRotationMatrix2D((c[0], c[1]), angle, 1)
    dst_img = cv2.warpAffine(img, M, (cols, rows))
    r, theta = cv2.cartToPolar(dst_point[:, 0], dst_point[:, 1], angleInDegrees=True)
    x, y = cv2.polarToCart(r, theta - angle, angleInDegrees=True)
    dst_point[:, 0] = x[:, 0] + c[0]
    dst_point[:, 1] = y[:, 0] + c[1]
    return dst_img, dst_point, dst_gaze


def zoom_img(img, point, c):  # done
    rows, cols = img.shape[:2]
    dst_point = point
    dst_point[:, 0] = dst_point[:, 0] - c[0]
    dst_point[:, 1] = dst_point[:, 1] - c[1]
    multiple = random.randint(100, 300)/100.
    M = cv2.getRotationMatrix2D((c[0], c[1]), 0, multiple)
    dst_img = cv2.warpAffine(img, M, (cols, rows))
    r, theta = cv2.cartToPolar(dst_point[:, 0], dst_point[:, 1], angleInDegrees=True)
    x, y = cv2.polarToCart(r * multiple, theta, angleInDegrees=True)
    dst_point[:, 0] = x[:, 0] + c[0]
    dst_point[:, 1] = y[:, 0] + c[1]
    return dst_img, dst_point


def switch(img, r=50):
    temp_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    temp_size = 2 * int(r) + 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, '%d' % temp_size, (10, 10), font,
                0.5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    while temp_size > 192:
        temp_size -= 30
    temp = cv2.resize(img_switch, (temp_size, temp_size))
    h, w = temp.shape[:2]
    res = cv2.matchTemplate(temp_img, temp, cv2.TM_CCOEFF_NORMED)
    # cv2.imshow('test1', res)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val > 0.45:
        left_top = max_loc
        right_bottom = (left_top[0] + w, left_top[1] + h)
        cv2.rectangle(img, left_top, right_bottom, 255, 2)


def thePMvalue(pointLT, pointRT, pointLB, pointRB):
    vector10 = pointLB - pointLT
    vector01 = pointRT - pointLT
    vector11 = pointRB - pointLT

    x11 = vector11[0]
    y11 = vector11[1]
    x10 = vector10[0]
    y10 = vector10[1]
    x01 = vector01[0]
    y01 = vector01[1]

    a0 = (y01 * x11 - x01 * y11) / (x10 * y01 - x01 * y10)
    a1 = (x11 - a0 * x10) / x01
    # print(vector01, vector10, vector11)
    print(pointLT, pointRT, pointLB, pointRB)
    if a1 != (y11 - a0 * y10) / y01:
        print("WA", a0, a1)

    return a0, a1, vector10, vector01


def PMY0Y1(vector10, vector01, Qx, Qy, pointLT):
    x10 = vector10[0]
    y10 = vector10[1]
    x01 = vector01[0]
    y01 = vector01[1]
    Qx = Qx - pointLT[0]
    Qy = Qy - pointLT[1]

    y0 = (y01 * Qx - x01 * Qy) / (x10 * y01 - x01 * y10)
    y1 = (Qx - y0 * x10) / x01

    return y0, y1


def PM(a0, a1, pointQx, pointQy, vector10, vector01, pointLT):  # Perspective Mapping
    y0, y1 = PMY0Y1(vector10, vector01, pointQx, pointQy, pointLT)
    denominator = a0 * a1 + a1*(a1 - 1) * y0 + a0*(a0 - 1) * y1
    x0 = (a1*(a0 + a1 - 1) * y0) / denominator
    x1 = (a0*(a0 + a1 - 1) * y1) / denominator
    # print(x0, x1, y0, y1, -pointQx, pointQy)
    pointRy = theround(x0) * screen_h
    pointRx = theround(x1) * screen_w
    return pointRx, pointRy


def theround(val):
    if val > 1.0:
        val = 1
    elif val < 0.0:
        val = 0
    return val


def two2three(n, m):
    x = -math.cos(n) * math.sin(m)
    y = -math.sin(n)
    z = -math.cos(n) * math.cos(m)
    return x, y, z

