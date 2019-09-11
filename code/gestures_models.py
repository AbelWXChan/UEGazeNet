import numpy as np
import cv2
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import time


class Model(object):
    def __init__(self):
        self._name = ''
        self._load_model = None
        self._save_model = None
        self._splits = 5
        self._repeats = 5
        self._data_path = 'gesture_data/all/'
        self._data_x = []
        self._data_y = []
        self._each_pattern = 300

    def data_sources(self):
        for i in range(17):
            file_path = self._data_path + '%d/' % i
            for j in range(self._each_pattern):
                img_L = file_path + '%d_L.jpg' % j
                img_R = file_path + '%d_R.jpg' % j
                # print(j)
                img = cv2.resize(cv2.imread(img_L, 0), (32, 32), interpolation=cv2.INTER_AREA)
                # cv2.namedWindow('a', cv2.WINDOW_NORMAL)
                # cv2.imshow('a', img)
                # print(img)
                # cv2.waitKey(0)
                img = np.reshape(img, [32 * 32])
                self._data_x.append(img)
                label = i
                self._data_y.append(label)

                img = cv2.resize(cv2.imread(img_R, 0), (32, 32), interpolation=cv2.INTER_AREA)
                img = np.reshape(img, [32 * 32])
                self._data_x.append(img)
                label = i
                self._data_y.append(label)
            print('------ %d ------' % i)
        np.array(self._data_x)
        np.array(self._data_y)
        data = np.hstack([self._data_x, np.reshape(self._data_y, [-1, 1])])
        np.save('./gesture_data/data.npy', data)
        print('------ done ------')

    def acc(self, model, x, y):
        start = time.clock()
        pre_y = model.predict(x)
        print(time.clock()-start, '\n', y, '\n', pre_y)
        correct_prediction = np.equal(y, pre_y)
        accuracy = np.mean(correct_prediction)
        return accuracy

    def train_model(self, model_name, x, y, _x, _y): 

        print(x.shape, y.shape)
        model = model_name
        model.fit(x, y)
        accuracy = self.acc(model, _x, _y)

        print(accuracy, model)

        return model

    def op_model(self, name):
        x = np.load('./gesture_data/data.npy')
        np.random.shuffle(x)
        print(x.shape)
        _x = x[:8160, :]

        test_x = x[8160:, 0:32 * 32] / 255.
        test_y = x[8160:, 32 * 32:].reshape([-1])

        train_x = _x[:, 0:32 * 32] / 255.
        train_y = _x[:, 32 * 32:].reshape([-1])

        if name == "KNN":  # 47%
            model = self.train_model(KNeighborsClassifier(n_neighbors=1, leaf_size=50, weights='distance', n_jobs=-1), x=train_x, y=train_y, _x=test_x, _y=test_y)
            return model 

        elif name == "RF":  # 76%
            model = self.train_model(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=60, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=2, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=40, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False), x=train_x, y=train_y, _x=test_x, _y=test_y)
            return model 

        elif name == "SVM":  # 87%
            model = self.train_model(SVC(C=2.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False), x=train_x, y=train_y, _x=test_x, _y=test_y)
            return model

        elif name == "xgb":  # 80%
            model = self.train_model(xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=6, min_child_weight=1, missing=None, n_estimators=50, n_jobs=-1, nthread=-1, objective='multi:softprob', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, silent=True, subsample=1), x=train_x, y=train_y, _x=test_x, _y=test_y)
            return model


work = Model()
# work.data_sources()
work.op_model('xgb')
# work.train()
