from pandas import Series
import os
import cv2
import numpy as np
import glob

img_path = r'/Users/momo/Downloads/rgbd_dataset_freiburg2_large_no_loop/rgbd_dataset_freiburg2_large_no_loop/'
data_dict = {}

if __name__ == '__main__':
    imgs = glob.glob(img_path+'*.png')
    for step, x in enumerate(sorted(imgs)):
        img = cv2.imread(os.path.join(x), 0)
        img = cv2.resize(img, (640, 480))
#        img = np.asarray(img)
        print(img.shape)
        corners = cv2.goodFeaturesToTrack(img, 1000, 0.001, 10)
        try:
            corners = corners.tolist()
            print(len(corners))
            string = ''
            for i in corners:
                img1 = cv2.rectangle(img, (int(i[0][0]) - 2, int(i[0][1]) - 2), (int(i[0][0]) + 2, int(i[0][1]) + 2), (0, 255, 0), 1)

                string += str(i[0][0])+' '+str(i[0][1])+' '
            data_dict[os.path.split(x)[-1]] = string
#            cv2.imwrite('my.png', img1)
            print(i)
        except AttributeError:
            print('nan')
            data_dict[x] = np.nan
        obj = Series(data_dict)
        obj.to_csv('rgbd_dataset_freiburg2_large_no_loop.csv', mode='a', header=False)
        print(x, 'complete!')
        data_dict = {}

