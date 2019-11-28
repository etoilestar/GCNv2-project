import torch
from torch.autograd import Function
import torch.nn as nn

class BinaryLayer(Function):
    @staticmethod
    def forward(ctx, input):
        result = torch.sign(input)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.clamp_(-1, 1)
        return grad_output


class GCNv2(torch.nn.Module):
    def __init__(self):
        super(GCNv2, self).__init__()
        self.elu = torch.nn.ELU(inplace=True)

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)

        self.conv3_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)

        self.conv4_1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)

        # Descriptor
        self.convF_1 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.convF_2 = torch.nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0)
#        self.convF_2 = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0) 
        # Detector
        self.convD_1 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.convD_2 = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.pixel_shuffle = torch.nn.PixelShuffle(16)
 
    def forward(self, x):
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))

        x = self.elu(self.conv3_1(x))
        x = self.elu(self.conv3_2(x))

        x = self.elu(self.conv4_1(x))
        x = self.elu(self.conv4_2(x))

        # Descriptor xF
        xF = self.elu(self.convF_1(x))
        desc = self.convF_2(xF)
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

        # Detector xD
        xD = self.elu(self.convD_1(x))
        det = self.convD_2(xD)#.sigmoid()
        det = self.pixel_shuffle(det)

        return desc, det


def later_deal(label, desc):
    Binary = BinaryLayer.apply
    """对每个batch的每个channel，分别进行grid_sample的操作，对于groundtruth中的关键点，找到featuremap中的对应位置，提取特征向量"""
    x, y = torch.nonzero(label[0] >= 1.0)[:, 0], torch.nonzero(label[0] >= 1.0)[:, 1]
#            如果为pytorch2.0版本可以改为使用下面两行语句其中一条代替上面语句
#            x, y = torch.nonzero(label[batch][0] == 1.0, as_tuple=True)
#            x, y = torch.where(label[batch][0] == 1.0)
    value = torch.gather(label[0][x.unsqueeze_(-1)].squeeze(), 1, y.unsqueeze_(-1))
    x = x.type(torch.FloatTensor)
    y = y.type(torch.FloatTensor)
    m = torch.cat((torch.cat((x.cuda(),y.cuda()), -1), value.cuda()), -1)
    
    m = m[m[:,2].argsort()]
    x, y = m[:, 0], m[:, 1]
    x.unsqueeze_(-1).cuda()
    y.unsqueeze_(-1).cuda()
    x_div = x*2/label.size()[-2]-1
    y_div = y*2/label.size()[-1]-1
    grid = torch.cat((x_div,y_div), -1).unsqueeze_(0).unsqueeze_(0).cuda()
    indice0 = torch.tensor(0).cuda()
    input_grid = torch.index_select(desc, index=indice0, dim=1)
    out_desc = torch.nn.functional.grid_sample(input_grid, grid)
    for i in range(1, desc.size()[1]):
        indice2 = torch.tensor(i).cuda()
        chip = torch.index_select(desc, index=indice2, dim=1)
        out_desc = torch.cat((out_desc, torch.nn.functional.grid_sample(chip, grid)), 0)

    out_desc = Binary(out_desc)
    out_desc.squeeze_()
    return out_desc, x, y


def GCNnet():
    return GCNv2()
