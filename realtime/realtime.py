import os
import cv2
import argparse
import torch.nn.parallel
from torch.utils.data import DataLoader

from module_2d import *
from dataLoader import loader
from vis_tool import draw_pose
from generateFeature import GFM

hands172msra = [0,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20,1,6,7,8]

def load_part_model(net, model_dir):
    pretrained_dict = torch.load(os.path.join(model_dir), map_location=lambda storage, loc: storage)
    model_dict = net.state_dict()
    for name, param in pretrained_dict.items():
        if name not in model_dict:
            continue
        model_dict[name].copy_(param)


class Trainer(object):
    def __init__(self, config):
        self.model_dir = config.model_dir
        self.data_dir = config.data_dir
        self.test_dataset = config.test_dataset
        self.train_dataset = config.train_dataset
        self.gpu_id = config.gpu_id

        # net para
        self.G_type = config.G_type
        self.joint_num = config.joint_num

        # load image para
        self.cube_size = config.cube_size
        self.input_size = config.input_size

        # multi_net
        self.feature_type = config.feature_type
        self.feature_sum = config.feature_sum
        self.dim_accumulate = config.dim_accumulate
        self.stage_type = config.stage_type

        self.deconv_size = config.deconv_size
        self.feature_name_list = self.feature_type.split(',')

        # use GPU
        self.cuda = torch.cuda.is_available()
        torch.cuda.set_device(self.gpu_id)

        dataset_name = self.data_dir.split('/')
        model_name = self.model_dir.split('/')
        model_name = model_name[-1].split('.')[0]
        self.save_dir = './result/'+dataset_name[-2]+'_'+dataset_name[-1] + '_' + self.train_dataset + '_' + model_name +'/'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.G = multi_stage(config)
        print(self.G)
        if self.test_dataset == 'kinect2':
            self.testData = loader.realtime_loader(self.data_dir, (363.9, 363.9, 255.4, 206.3),
                                        1500, cube_size=self.cube_size)
        elif self.test_dataset == 'realsense':
            self.testData = loader.realtime_loader(self.data_dir, (615.866, 615.866, 316.584, 228.38),
                                        1500, cube_size=self.cube_size)

        self.testLoader = DataLoader(self.testData, batch_size=1, shuffle=False, num_workers=1)
        self.G = self.G.cuda()
        self.G.load_state_dict(torch.load(os.path.join(self.model_dir), map_location=lambda storage, loc: storage))
        self.GFM_ = GFM()

    def run(self):
        self.G.eval()
        filename = self.save_dir + '/joint.txt'
        file = open(filename, 'w')
        for batch_idx, data in enumerate(self.testLoader):
            img, center, center_uvd, M, cube = data
            img = img.cuda()
            with torch.no_grad():

                outputs, features = self.G(img, self.GFM_, self.testData, M, cube, center)
                # output = outputs[-1]
                for index in range(len(outputs)):
                    # post process
                    output = outputs[index]
                    output = output.view(1, -1, 3)
                    joints_xyz = self.testData.uvd_nl2xyznl_tensor(output, M, cube, center)
                    joints_uvd = output

                    joints_draw = (joints_uvd.detach().cpu().numpy() + 1) * (self.input_size / 2)
                    img_draw = (img.cpu().numpy() + 1) / 2 * 255

                    hands172msra = [0, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20, 1, 6, 7, 8]
                    joints_xyz = \
                    (joints_xyz * cube.cuda().view(1, 1, 3) / 2 + center.cuda().view(1, 1, 3)).cpu().numpy()[0]
                    np.savetxt(file, joints_xyz[hands172msra, :].reshape(1, -1), fmt='%.3f')
                    img_dir = self.save_dir + str(batch_idx) +'_'+str(index)+ '.png'
                    img_show = draw_pose(args.train_dataset, cv2.cvtColor(img_draw[0, 0], cv2.COLOR_GRAY2RGB),
                                         joints_draw[0])
                    cv2.imwrite(img_dir, img_show)
                print(batch_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/kinect2')
    parser.add_argument('--model_dir', type=str, default='./checkpoint/hands17/offset_000_240.pth')
    parser.add_argument('--train_dataset', default='hands17', type=str, choices=['nyu', 'icvl', 'msra','hands17','itop'])
    parser.add_argument('--test_dataset', default='kinect2', type=str, choices=['kinect2','realsense'])

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--input_size', default=128, type=int)
    parser.add_argument('--cube_size', default=[250, 250, 250], type=int)
    parser.add_argument('--joint_num', default=21, type=int) # nyu 14 icvl 16 msra 21 hands17 21

    parser.add_argument('--G_type', default='multi_net_resnet18', type=str)
    parser.add_argument('--feature_type', default='offset', type=str, choices=['3Dheatmap','heatmap_nodepth','heatmap','offset', 'joint_decode', 'feature_upsample','heatmap_cat' 'edt', 'GD', 'heatmap3D','point_heatmap', 'point_joint', 'heatmap_multiview'])
    parser.add_argument('--dim_accumulate', default=True, type=bool)
    parser.add_argument('--deconv_size', default=64, type=int)
    parser.add_argument('--heatmap_std', default=3.4, type=float)
    parser.add_argument('--stage_type', default=[0, 0, 0], type=int)# 0 regression 1 handmodel 2detection
    parser.add_argument('--feature_sum', default=False, type=bool)

    parser.add_argument('--pool_factor', default=4, type=int)

    args = parser.parse_args()

    predictor = Trainer(args)

    predictor.run()




