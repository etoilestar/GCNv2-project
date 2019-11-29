import torch
import glob
import cv2
from network.GCNv2 import GCNnet
from config import params

def deal(img):
#    img = cv2.imread(img, 0)
    array = cv2.resize(img, (640,480))
 #   array = np.array(array)
    img = array/255.0
    return img.astype('float32'), array

if __name__ == '__main__':
    net = GCNnet()

    net = net.cuda(params['gpu'][0])
    net = torch.nn.DataParallel(net, device_ids=params['gpu'])
    test_model = 'GCNv2_2.pth'
    model_dict = torch.load(test_model, map_location='cpu')
    net.load_state_dict(model_dict)
    print('load',test_model,'successfully' )
    video_name = 'my.mp4'
    cap = cv2.VideoCapture(video_name)
    ret, frame = cap.read()
    video = cv2.VideoWriter("VideoTest1.mp4", cv2.VideoWriter_fourcc('I', '4', '2', '0'), 24, (640, 480))
    while ret:
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        input, array = deal(img)
        input = torch.from_numpy(input).cuda()
        input.unsqueeze_(0).unsqueeze_(0)
        with torch.no_grad():
            _, out = net(input)
            out = out.sigmoid()
        key_point = torch.nonzero(out>0.5)
#        print(array.shape[0])
        for p in key_point:
            u = p.cpu().numpy()[-1]
            v = p.cpu().numpy()[-2]
            assert u<= array.shape[1] and v<=array.shape[0]
            array = cv2.rectangle(array, (u - 2, v - 2), (u + 2, v + 2), (0, 255, 0), 1)
        video.write(array)
    video.release()