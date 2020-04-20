import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np


from resnet import Bottleneck,BasicBlock


"""
loss function
"""
class Modified_SmoothL1Loss(torch.nn.Module):

    def __init__(self):
        super(Modified_SmoothL1Loss,self).__init__()

    def forward(self,x,y):
        total_loss = 0
        assert(x.shape == y.shape)
        z = (x - y).float()
        mse = (torch.abs(z) < 0.01).float() * z
        l1 = (torch.abs(z) >= 0.01).float() * z
        total_loss += torch.sum(self._calculate_MSE(mse))
        total_loss += torch.sum(self._calculate_L1(l1))

        return total_loss/z.shape[0]

    def _calculate_MSE(self, z):
        return 0.5 *(torch.pow(z,2))

    def _calculate_L1(self,z):
        return 0.01 * (torch.abs(z) - 0.005)

class coord_weight_SmoothL1Loss(torch.nn.Module):

    def __init__(self,coord_weight=[1.0,1.0,1.0]):
        super(coord_weight_SmoothL1Loss,self).__init__()
        self.weight = torch.from_numpy(np.array(coord_weight)).float().view(1,1,3)
    def forward(self,x,y):
        total_loss = 0
        device = x.device
        assert(x.shape == y.shape)
        x = x.view(x.size(0),-1,3)
        y = y.view(y.size(0),-1,3)
        z = (x - y).float()
        self.weight = self.weight.to(device)
        mse = (torch.abs(z) < 0.01).float() * z
        l1 = (torch.abs(z) >= 0.01).float() * z
        mse_loss = torch.sum(torch.sum(self._calculate_MSE(mse)* self.weight,dim=0), dim=0)
        l1_loss = torch.sum(torch.sum(self._calculate_L1(l1)* self.weight,dim=0), dim=0)

        total_loss += torch.sum(mse_loss)
        total_loss += torch.sum(l1_loss)

        return total_loss/z.shape[0]

    def _calculate_MSE(self, z):
        return 0.5 *(torch.pow(z,2))

    def _calculate_L1(self,z):
        return 0.01 * (torch.abs(z) - 0.005)


# MD_:feature map block
class FMB_(nn.Module):
    def __init__(self, inchannels):
        super(FMB_, self).__init__()
        self.uppool0 = nn.ConvTranspose2d(inchannels, inchannels/2, kernel_size=2, stride=2)
        self.relu0 = nn.ReLU()
        self.uppool1 = nn.ConvTranspose2d(inchannels/2, inchannels/4, kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.uppool0(x)
        out = self.relu0(out)
        out = self.uppool1(out)
        out = self.relu1(out)
        return out


# MD_:linear map block
class LMB_(nn.Module):
    def __init__(self, joint_num, outchannels, outsize):
        super(LMB_, self).__init__()
        self.fc1 = nn.Linear(joint_num*3, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, outchannels*outsize*outsize)
        self.outchannels = outchannels
        self.outsize = outsize
        self.joint_num = joint_num

    def forward(self, x):
        x = x.view(-1, self.joint_num*3)
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.view(-1, self.outchannels, self.outsize, self.outsize)
        return out


# MD_:linear relu map module
class LRMB_(nn.Module):
    def __init__(self, joint_num, outchannels, outsize):
        super(LRMB_, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(joint_num * 3, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, outchannels * outsize * outsize)
        self.outchannels = outchannels
        self.outsize = outsize
        self.joint_num = joint_num

    def forward(self, x):
        x = x.view(-1, self.joint_num*3)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = out.view(-1, self.outchannels, self.outsize, self.outsize)
        return out


class MyReLU(torch.autograd.Function):

    def forward(self, input_):

        self.save_for_backward(input_)
        output = input_.clamp(min=0)
        return output

    def backward(self, grad_output):

        input_, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class multi_stage(nn.Module):
    def __init__(self, args):
        super(multi_stage, self).__init__()
        self.dataset = args.train_dataset
        self.joint_num = args.joint_num
        self.stage_type = args.stage_type
        self.feature_type = args.feature_type
        self.feature_size = int(args.input_size / args.pool_factor)
        self.dim_accumulate = args.dim_accumulate
        self.inplanes = 64
        self.BN_MOMENTUM = 0.1
        self.deconv_with_bias = False

        if 'resnet18' in args.G_type:
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif 'resnet50' in args.G_type:
            block = Bottleneck
            layers = [3, 4, 6, 3]

        if args.feature_type =='heatmap_nodepth':
            self.add_dim = self.joint_num
        elif args.feature_type =='heatmap':
            self.add_dim = self.joint_num*2
        elif 'decode' in args.feature_type:
            self.add_dim = 64
        elif 'offset' in args.feature_type:
            self.add_dim = self.joint_num * 4
        elif '3Dheatmap' in args.feature_type:
            self.add_dim = self.joint_num
        elif 'feature_upsample' == args.feature_type:
            self.add_dim =64
            self.feature_net = nn.Sequential(
                nn.Upsample(scale_factor=(8,8)),
                nn.Conv2d(512*block.expansion,64,kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _make_layer(block, 64, 64, layers[0], stride=1)
        )

        self.fcs = nn.ModuleList()
        self.features = nn.ModuleList()
        self.avg_pool = nn.AvgPool2d(4, stride=1)
        self.dim = 0
        for index,type in enumerate(args.stage_type):
            # stacked regression-based block
            if type == 0:
                self.features.append(
                    nn.Sequential(
                        _make_layer(block, 64 * block.expansion + self.dim, 128, layers[1], stride=2),
                        _make_layer(block, 128 * block.expansion, 256, layers[2], stride=2),
                        _make_layer(block, 256 * block.expansion, 512, layers[3], stride=2),
                    )
                )
                if self.dim_accumulate:
                    self.dim = self.dim + self.add_dim
                else:
                    self.dim = self.add_dim
                self.fcs.append(nn.Linear(512*block.expansion, self.joint_num * 3))
            # stacked detection-based block
            elif type == 2:
                deconv_num = 3
                deconv_dim = []
                deconv_k = []
                self.inplanes = 512
                for i in range(deconv_num):
                    deconv_dim.append(256)
                    deconv_k.append(4)
                self.features.append(
                    nn.Sequential(
                        _make_layer(block, 64 * block.expansion + self.dim, 128, layers[1], stride=2),
                        _make_layer(block, 128 * block.expansion, 256, layers[2], stride=2),
                        _make_layer(block, 256 * block.expansion, 512, layers[3], stride=2),
                        self._make_deconv_layer(deconv_num, deconv_dim, deconv_k),
                        nn.Conv2d(in_channels=256, out_channels=self.joint_num*4, kernel_size=1, stride=1, padding=0)
                    )
                )

    def forward(self, img, GFM_, loader, M=None, cube=None, center=None, decode_net=None):
        device = img.device
        feature = self.pre(img)
        remap_feature = torch.Tensor().to(device)
        pos_list = []
        remap_feature_list = []
        for (i, type)in enumerate(self.stage_type):
            c5 = self.features[i](torch.cat((feature,remap_feature),dim=1))
            if type == 0:
                y = self.avg_pool(c5)
                y = self.fcs[i](y.view(y.size(0), -1))
                y = y.view(y.size(0), -1, 3)
            elif type == 1:
                y = self.avg_pool(c5)
                y = self.fcs[i](y.view(y.size(0), -1))
                y = self.handmodelLayer.calculate_position(y).view(y.size(0), -1, 3)
            elif type == 2:
                y = GFM_.offset2joint(c5, img)
                y = y.view(y.size(0), -1, 3)
            pos_list.append(y)
            feature_temp = self.repara_module(img, y, c5, GFM_, loader, M, cube, center, decode_net=decode_net)
            if self.dim_accumulate:
                remap_feature = torch.cat((remap_feature, feature_temp),dim=1)
            else:
                remap_feature = feature_temp
            remap_feature_list.append(remap_feature)

        return pos_list,remap_feature_list

    def repara_module(self, img, pos, c5, GFM_,loader,  M, cube, center, decode_net=None):
        if self.feature_type == 'heatmap':
            heatmap = GFM_.joint2heatmap2d(pos, isFlip=False)
            depth = heatmap * pos[:, :, 2].view(pos.size(0), -1, 1, 1)
            feature = torch.cat((heatmap,depth),dim=1)
        elif self.feature_type == 'heatmap_nodepth':
            heatmap = GFM_.joint2heatmap2d(pos, isFlip=False)
            feature = heatmap
        elif self.feature_type == '3Dheatmap':
            pos_xyz = loader.uvd_nl2xyznl_tensor(pos, M, cube, center)
            feature = GFM_.joint2offset(pos_xyz, img, feature_size=self.feature_size)[:,self.joint_num*3:,:,:]
        elif self.feature_type == 'offset':
            feature = GFM_.joint2offset(pos, img, feature_size=self.feature_size)
        elif self.feature_type == 'joint_decode':
            feature = decode_net(pos)
        elif self.feature_type == 'offset_decode':
            offset = GFM_.joint2offset(pos, img, feature_size=self.feature_size)
            feature = decode_net(offset)
        elif self.feature_type == 'feature_upsample':
            feature = self.feature_net(c5)
        return feature

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def init_weights(self, pretrained=''):
        if True:
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)


def _make_layer(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


def _make_heatmap_layer(block, inplanes, planes, blocks, stride=1,heatmap_dim=1):
    downsample = False
    if stride != 1 or inplanes != planes * block.expansion:
        downsample=True
    layers = []
    layers.append(block(inplanes, planes, stride, downsample,heatmap_dim=heatmap_dim))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes,heatmap_dim=heatmap_dim))

    return nn.Sequential(*layers)

