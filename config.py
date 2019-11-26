params = dict()

params['data'] = '~/Downloads/rgbd_dataset_freiburg3_long_office_household/rgb/1341847989.334871.png'
params['pretrained'] = False#'./model/GCNv2.pth'
params['save_path'] = './save/1341847989.334871.png'
params['imagepath'] = '../img/'
params['keypoint'] = './data_keypoint/'
params['gt'] = './gt/'
params['batch_size'] = 32
params['gpu'] = [0]
params['c'] = 8
params['learning_rate'] = 1e-4
params['num_epoch'] = 300
params['model_path'] = './model/GCNv2.pth'
params['random drop'] = 0.9
params['cal_match'] = True
params['log']  = './log/loss.txt'
