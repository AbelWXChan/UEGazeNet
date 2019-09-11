import os
import shutil

for pp in range(15):
    target = 'test_data/%d/' % pp
    dst = 'test_data/all/'
    first, second = os.listdir(target)


    def dst_exist(path):
        if os.path.exists(path) is False:
            os.makedirs(path)


    def work(dir_path):
        dir_list = os.listdir(target + dir_path)
        for i in dir_list:
            file_list_a = target + dir_path + '/%s/' % i
            try:
                file_list_b = os.listdir(dst + '/%s' % i)
            except:
                dst_exist(dst + '/%s' % i)
                file_list_b = os.listdir(dst + '/%s' % i)
            num = len(file_list_b)//2
            for j in range(10):
                img_L = file_list_a + '%d_L.jpg' % j
                img_R = file_list_a + '%d_R.jpg' % j
                # txt_L = file_list_a + '%d_L.txt' % j
                # txt_R = file_list_a + '%d_R.txt' % j
                shutil.copyfile(img_L, dst + '/%s' % i + '/%d_L.jpg' % (num + j))
                shutil.copyfile(img_R, dst + '/%s' % i + '/%d_R.jpg' % (num + j))
                # shutil.copyfile(txt_L, dst + '/%s' % i + '/%d_L.txt' % (num + j))
                # shutil.copyfile(txt_R, dst + '/%s' % i + '/%d_R.txt' % (num + j))
            print(i, num+10)
        print('------- done -------')


    work(first)
    work(second)
