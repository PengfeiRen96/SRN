import os
import cv2
import argparse
import torch.nn.parallel
from torch.utils.data import DataLoader

from module_2d import *
from dataLoader import train_loader
from vis_tool import debug_2d_pose
from generateFeature import GFM

from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

class Trainer(object):
    def __init__(self, config):
        self.test_during_train = True
        # train para
        self.summary_name = config.summary_name
        self.model_dir = config.model_dir
        self.root_dir = config.root_dir
        self.data_dir = self.root_dir+'/'+config.dataset+'/'
        self.dataset = config.dataset
        self.gpu_id = config.gpu_id
        self.bacth_size = config.bacth_size
        self.lr = config.lr
        self.opt = config.opt
        self.max_epoch = config.epoch
        self.step_epoch = config.step

        # net para
        self.G_type = config.G_type
        self.joint_num = config.joint_num

        # load image para
        self.cube_size = config.cube_size
        self.input_size = config.input_size
        self.center_type = config.center_type
        self.aug_para = config.aug_para

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

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.model_dir+'/debug/'):
            os.mkdir(self.model_dir+'/debug/')

        self.writer = SummaryWriter('./runs/' + self.dataset + '-' + self.summary_name)

        # define network
        self.net = multi_stage(config)
        print(self.net)
        self.net = self.net.cuda()
        self.Loss = Modified_SmoothL1Loss().cuda()
        self.GFM_ = GFM()

        # load data
        if self.dataset == 'nyu':
            self.trainDataset = train_loader.nyu_loader(self.data_dir, 'train', center_type=self.center_type,
                                                        aug_para=self.aug_para, img_size=self.input_size, cube_size=self.cube_size)
            self.testDataset = train_loader.nyu_loader(self.data_dir, 'test', center_type=self.center_type,
                                                       img_size=self.input_size, cube_size=self.cube_size)
        if self.dataset == 'icvl':
            self.trainDataset = train_loader.icvl_loader(self.data_dir, 'train', full_img=True, center_type=self.center_type,
                                                        aug_para=self.aug_para, img_size=self.input_size)
            self.testDataset = train_loader.icvl_loader(self.data_dir, 'test', center_type=self.center_type,
                                                       img_size=self.input_size)
        if self.dataset == 'msra':
            self.trainDataset = train_loader.msra_loader(self.data_dir, 'train', center_type=self.center_type,
                                                        aug_para=self.aug_para, img_size=self.input_size)
            self.testDataset = train_loader.msra_loader(self.data_dir, 'test', center_type=self.center_type,
                                                       img_size=self.input_size)
        if self.dataset == 'hands17':
            self.trainDataset = train_loader.hands17_loader(self.data_dir, 'train', center_type=self.center_type,
                                                        aug_para=self.aug_para, img_size=self.input_size, cube_size=self.cube_size)
            self.testDataset = train_loader.hands17_loader(self.data_dir, 'test', center_type=self.center_type,
                                                       img_size=self.input_size, cube_size=self.cube_size)

        self.trainLoader = DataLoader(self.trainDataset, batch_size=self.bacth_size, shuffle=True, num_workers=4)
        self.testLoader = DataLoader(self.testDataset, batch_size=self.bacth_size, shuffle=False, num_workers=4)
        print(self.trainDataset.__len__())
        # define opt
        optimList = [{"params": self.net.parameters(), "initial_lr": self.lr}]
        if self.opt == 'sgd':
            self.optimizer = SGD(optimList, lr=self.lr, momentum=0.9, weight_decay=1e-4)
        elif self.opt == 'adam':
            self.optimizer = Adam(optimList, lr=self.lr)#weight_decay=0.00005
        else:
            raise Exception("undefine")

        self.scheduler = StepLR(self.optimizer, step_size=self.step_epoch, gamma=0.1)
        self.min_error = 1000

    def train(self):
        self.phase = 'train'

        # train
        for epoch in range(0, self.max_epoch):
            self.net.train()

            for ii, data in tqdm(enumerate(self.trainLoader)):
                joint_xyz_list = []
                joint_uvd_list = []
                img, xyz_gt, uvd_gt, center, M, cube = data
                img, uvd_gt, xyz_gt = img.cuda(), uvd_gt.cuda(), xyz_gt.cuda()
                center, M, cube = center.cuda(), M.cuda(), cube.cuda()

                self.optimizer.zero_grad()

                outputs, features = self.net(img, self.GFM_, self.trainDataset)
                loss = 0
                for stage_index, stage_type in enumerate(self.stage_type):
                    # regress
                    if stage_type == 0:
                        joints_uvd = outputs[stage_index]
                        joints_xyz = self.trainDataset.uvd_nl2xyznl_tensor(joints_uvd, center, M, cube)
                        loss_coord = self.Loss(joints_uvd, uvd_gt)
                        loss += loss_coord

                    joint_xyz_list.append(joints_xyz)
                    joint_uvd_list.append(joints_uvd)
                    batch_joint_error = self.xyz2error(joints_xyz, xyz_gt, center, cube)
                    error = np.mean(batch_joint_error)
                    self.writer.add_scalar('loss' + str(stage_index), loss_coord.item(), epoch * len(self.trainLoader) + ii)
                    self.writer.add_scalar('error' + str(stage_index), error.item(), epoch * len(self.trainLoader) + ii)

                # update para
                loss.backward()
                self.optimizer.step()

                if ii % 1000 == 0:
                    img_label = debug_2d_pose(img, uvd_gt, ii, self.dataset, self.model_dir + '/debug/', 'label')
                    self.writer.add_image('label', np.transpose(img_label, (2, 0, 1))/255.0, epoch * len(self.trainLoader) + ii)
                    for img_index, stage_type in enumerate(self.stage_type):
                        img_predict = debug_2d_pose(img, joint_uvd_list[img_index], ii, self.dataset, self.model_dir + '/debug/', 'predict')
                        self.writer.add_image('predict'+str(img_index), np.transpose(img_predict, (2, 0, 1))/255.0, epoch * len(self.trainLoader) + ii)

            save = {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(
                save,
                self.model_dir + "/latest.pth"
            )

            if self.test_during_train:
                test_error = self.test(epoch)
                if test_error <= self.min_error:
                    self.min_error = test_error
                    save = {
                        "model": self.net.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch
                    }
                    torch.save(
                        save,
                        self.model_dir + "/best.pth"
                    )

            self.scheduler.step(epoch)

    @torch.no_grad()
    def test(self, epoch=-1):
        '''
        计算模型测试集上的准确率
        '''

        self.result_file_list = []
        for file_index in range(len(self.stage_type)):
            self.result_file_list.append(open(self.model_dir+'/result_'+str(file_index)+'.txt', 'w'))

        self.phase = 'test'
        self.net.eval()

        Error = np.zeros([len(self.stage_type)])
        batch_num = 0

        for ii, data in tqdm(enumerate(self.testLoader)):
            joint_xyz_list = []
            joint_uvd_list = []
            img, xyz_gt, uvd_gt, center, M, cube = data
            img, uvd_gt, xyz_gt = img.cuda(), uvd_gt.cuda(), xyz_gt.cuda()
            center, M, cube = center.cuda(), M.cuda(), cube.cuda()

            outputs, features = self.net(img, self.GFM_, self.trainDataset)
            batch_num += 1

            for stage_index, stage_type in enumerate(self.stage_type):
                if stage_type == 0:
                    joints_uvd = outputs[stage_index]
                    joints_xyz = self.testDataset.uvd_nl2xyznl_tensor(joints_uvd, center, M, cube)

                joint_uvd_list.append(joints_uvd)
                joint_xyz_list.append(joints_xyz)
                joint_errors = self.xyz2error(joints_xyz, xyz_gt, center, cube, write_file=True, stage_index=stage_index)
                error = np.mean(joint_errors)
                Error[stage_index] += error

        mean_Error = Error / batch_num
        for error_index in range(mean_Error.shape[0]):
            self.writer.add_scalar("test_error" + str(error_index), mean_Error[error_index], epoch)
        # error_info = ''
        # for error_index in range(mean_Error.shape[0]):
        #     # self.writer.add_scalar("test/mean_Error"+str(error_index), mean_Error[error_index], epoch)
        #     print(
        #         "[mean_Error %.5f]" % (mean_Error[error_index])
        #     )
        #     error_info += ' error' + str(error_index) + ": %.5f" % (mean_Error[error_index]) + ' '

        return min(mean_Error)

    @torch.no_grad()
    def xyz2error(self, output, joint, center, cube_size, write_file=False, stage_index=0):
        output = output.detach().cpu().numpy()
        joint = joint.detach().cpu().numpy()
        center = center.detach().cpu().numpy()
        cube_size = cube_size.detach().cpu().numpy()
        batchsize, joint_num, _ = output.shape
        center = np.tile(center.reshape(batchsize, 1, -1), [1, joint_num, 1])
        cube_size = np.tile(cube_size.reshape(batchsize, 1, -1), [1, joint_num, 1])

        joint_xyz = output * cube_size / 2 + center
        joint_world_select = joint * cube_size / 2 + center

        errors = (joint_xyz - joint_world_select) * (joint_xyz - joint_world_select)
        errors = np.sqrt(np.sum(errors, axis=2))

        if self.phase == 'test' and write_file:
            np.savetxt(self.result_file_list[stage_index], self.testDataset.joint3DToImg(joint_xyz).reshape([batchsize, joint_num * 3]), fmt='%.3f')

        return errors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_name', type=str, default='test')
    parser.add_argument('--model_dir', type=str, default='./test')
    # parser.add_argument('--root_dir', type=str, default='D:\\dataset\\')
    parser.add_argument('--root_dir', type=str, default='/home/pfren/dataset/hand/')

    parser.add_argument('--dataset', default='hands17', type=str, choices=['nyu', 'icvl', 'msra','hands17'])
    parser.add_argument('--joint_num', default=21, type=int) # nyu 14 icvl 16 msra 21 hands17 21
    parser.add_argument('--input_size', default=128, type=int)
    parser.add_argument('--cube_size', default=[200, 200, 200], type=int)
    parser.add_argument('--center_type', default='refine', type=str, choices=['refine', 'joint_mean', 'mass'])
    parser.add_argument('--aug_para', default=[10, 0.2, 180])

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--bacth_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--lr', type=int, default=0.3)
    parser.add_argument('--opt', type=str, default='sgd', choices=['sgd', 'adam'])

    parser.add_argument('--G_type', default='multi_net_resnet18', type=str)
    parser.add_argument('--feature_type', default='offset', type=str, choices=['3Dheatmap','heatmap_nodepth','heatmap','offset', 'joint_decode', 'feature_upsample','heatmap_cat' 'edt', 'GD', 'heatmap3D','point_heatmap', 'point_joint', 'heatmap_multiview'])
    parser.add_argument('--dim_accumulate', default=True, type=bool)
    parser.add_argument('--deconv_size', default=64, type=int)
    parser.add_argument('--heatmap_std', default=3.4, type=float)
    parser.add_argument('--stage_type', default=[0, 0], type=int)# 0 regression 1 handmodel 2detection
    parser.add_argument('--feature_sum', default=False, type=bool)

    parser.add_argument('--pool_factor', default=4, type=int)

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()




