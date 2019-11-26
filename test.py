import torch
import glob
import cv2
from network.GCNv2 import GCNnet
from config import params

def deal(img):
    img = cv2.imread(img, 0)
    array = cv2.resize(img, (640,480))
 #   array = np.array(array)
    img = array/255.0
    return img.astype('float32'), array

if __name__ == '__main__':
    net = GCNnet()

    net = net.cuda(params['gpu'][0])
    net = torch.nn.DataParallel(net, device_ids=params['gpu'])
    net = torch.nn.DataParallel(net, device_ids=params['gpu'])
    test_model = params['model_path']
    model_dict = torch.load(test_model, map_location='cpu')
#    net_dict = net.state_dict()
#    net_dict.update(model_dict)
    net.load_state_dict(model_dict)
#    print('load',params['model_path'],'successfully' )
#    net = net.cuda(params['gpu'][0])
#    net = torch.nn.DataParallel(net, device_ids=params['gpu'])
#    loss = torch.nn.BCEWithLogitsLoss(pos_weight=w)
    img_path = './test_img/'
    save_path = './save_test/'
#    imgs = glob.glob('./test_img/1.png')
    imgs = glob.glob('../img/rgbd_dataset_freiburg3_long_office_household/1341847994.065706.png')
    for i , img in enumerate(sorted(imgs)):
        input, array = deal(img)
        input = torch.from_numpy(input).cuda()
        input.unsqueeze_(0).unsqueeze_(0)
        with torch.no_grad():
            _, out = net(input)
            out = out.sigmoid()
        # out = torch.where(out>0.5, 1.0, 0.0)
        print(out.size(), torch.unique(out))
        key_point = torch.nonzero(out>0.5)
        print(key_point.size()[0])
#        print(array.shape[0])
        for p in key_point:
            u = p.cpu().numpy()[-1]
            v = p.cpu().numpy()[-2]
            assert u<= array.shape[1] and v<=array.shape[0]
            array = cv2.rectangle(array, (u - 2, v - 2), (u + 2, v + 2), (0, 255, 0), 1)
            cv2.imwrite(save_path+str(i)+'.png', array)
