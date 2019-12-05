import torch
import glob
import cv2
from network.GCNv2 import GCNnet
import numpy as np
from config import params
from tqdm import tqdm

def deal(img):
    img = img/255.0
    return img.astype('float32')

def get_vector(x, y, desc, shape):
    x1 , y1 = x.astype('float32'),y.astype('float32')
    x_div = torch.tensor(x1 * 2 / shape[0] - 1).unsqueeze(-1)
    y_div = torch.tensor(y1 * 2 / shape[1] - 1).unsqueeze(-1)
    grid = torch.cat((x_div, y_div), -1).unsqueeze_(0).unsqueeze_(0).cuda(params['gpu'][0])
    indice0 = torch.tensor(0).cuda(params['gpu'][0])
    input_grid = torch.index_select(desc, index=indice0, dim=1)
    out_desc = torch.nn.functional.grid_sample(input_grid, grid)
    for i in range(1, desc.size()[1]):
        indice2 = torch.tensor(i).cuda(params['gpu'][0])
        chip = torch.index_select(desc, index=indice2, dim=1)
        out_desc = torch.cat((out_desc, torch.nn.functional.grid_sample(chip, grid)), 0)
    return out_desc,torch.tensor(x1), torch.tensor(y1)

def nms(key_point):
    nms_key_point = []
    kp_array = key_point.cpu().numpy()
    pts_raw = list()
    for p in key_point:
        u = int(p.cpu().numpy()[-2])
        v = int(p.cpu().numpy()[-3])
        #               image = cv2.rectangle(frame, (u - 2, v - 2), (u + 2, v + 2), (0, 255, 0), 1)
        pts_raw.append((u, v))

    grid = np.zeros([480, 640])
    inds = np.zeros([480, 640])

    for i in range(len(pts_raw)):
        uu = (int)(pts_raw[i][0])
        vv = (int)(pts_raw[i][1])

        grid[vv][uu] = 1
        inds[vv][uu] = i
    grid = cv2.copyMakeBorder(grid, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0)
    for i in range(len(pts_raw)):
        uu = (int)(pts_raw[i][0]) + 8
        vv = (int)(pts_raw[i][1]) + 8

        if grid[vv][uu] != 1:
            continue

        for k in range(-8, 8 + 1):
            for j in range(-8, 8 + 1):
                if j == 0 and k == 0:
                    continue

                grid[vv + k][uu + j] = 0

        grid[vv][uu] = 2

    valid_cnt = 0
#    print()

    for v in range(480 + 8):
        for u in range(640 + 8):
            if (u - 8 >= (640 - 16) or u - 8 < 16 or
                    v - 8 >= (480 - 16) or v - 8 < 16):
                continue

            if (grid[v][u] == 2):
                image = cv2.rectangle(frame, (u-8 - 2, v-8 - 2), (u-8 + 2, v-8 + 2), (0, 255, 0), 1)
                nms_key_point.append([u, v])
    nms_key_point = np.asarray(nms_key_point)
    return image, nms_key_point

if __name__ == '__main__':
    net = GCNnet()
    net = net.cuda(params['gpu'][0])
    net = torch.nn.DataParallel(net, device_ids=params['gpu'])
#    net = torch.nn.DataParallel(net, device_ids=params['gpu'])
    test_model = './model/GCNv2.pth'
    model_dict = torch.load(test_model, map_location='cpu')
    net.load_state_dict(model_dict)
    print('load',test_model,'successfully' )
    video_name = 't1.mp4'
    cap = cv2.VideoCapture(video_name)
    ret, frame = cap.read()
    video = cv2.VideoWriter("VideoTest1.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (640, 480))
    for i in tqdm(range(100)):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640,480))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        input = deal(img)
        input = torch.from_numpy(input).cuda(params['gpu'][0])
        input.unsqueeze_(0).unsqueeze_(0)
        with torch.no_grad():
            des, out = net(input)
            out = out.sigmoid()
        out.squeeze_()
        key_point = torch.nonzero(out>0.58)
#        print(key_point)
        key_point = torch.cat((key_point.float(), torch.gather(out[key_point[:,0]],1, key_point[:, 1].unsqueeze(-1))), -1)
        key_point = key_point[key_point[:,2].argsort(descending = True)]
#        print(key_point)
#        print(array.shape[0])
        image, nms_key_point = nms(key_point)
        des_vectors, x, y = get_vector(nms_key_point[:, 0], nms_key_point[:, 1], des, img.shape)
        des_vectors = torch.where(des_vectors>=0, torch.tensor(1.0).cuda(params['gpu'][0]), torch.tensor(-1.0).cuda(params['gpu'][0]))
        if i == 0:
            des_old = des_vectors
            x_old = x
            y_old = y
        else:
            cross_dis = torch.t(des_vectors.squeeze())@des_old.squeeze()
            sort_index = cross_dis.sort(1, descending=True).indices[:, :1]
#            print(cross_dis.size(),des_old.squeeze().size(), des_vectors.squeeze().size(), sort_index.size())
            sort_values = cross_dis.sort(1, descending=True).values[:, :1]
            sort_x2, sort_y2 = x_old[sort_index], y_old[sort_index]
            d_matrix = torch.sqrt(torch.pow((x - sort_x2), 2) + torch.pow((y - sort_y2),2))
            d = torch.mean(d_matrix)
            text = 'avg distance:%0.4f'%d.item() 
            text2 = 'avg feature distance:%0.4f'%(16-torch.mean(sort_values)/2).item()
#            print('avg location distance:%0.4f'%d.item(), 'avg feature distance:%0.4f'%((32-torch.mean(sort_values))/2).item())
            cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.4, (256, 0, 0), 1)
            cv2.putText(image, text2, (50, 70), cv2.FONT_HERSHEY_COMPLEX, 0.4, (256, 0, 0), 1)
            des_old = des_vectors
            x_old = x
            y_old = y
        video.write(image)
#        print('img{}'.format(i), 'finish!')
    video.release()
