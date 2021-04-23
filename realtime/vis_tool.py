import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from enum import Enum

calculate = [0,3,5,8,10,13,15,18,24,25,26,28,29,30]


def get_param(dataset):
    if dataset == 'icvl':
        return 240.99, 240.96, 160, 120
    elif dataset == 'nyu':
        return 240.99, 240.96, 160, 120
    elif dataset == 'nyu_full':
        return 240.99, 240.96, 160, 120
        # return 588.03, 587.07, 320, 240
    elif dataset == 'msra':
        return 241.42, 241.42, 160, 120
    elif dataset == 'hands17':
        return 475.065948, 475.065857, 315.944855, 245.287079
    elif dataset == 'xtion2':
        return 535.4, 539.2, 320.1, 247.6
    elif dataset == 'itop':
        return 285.71, 285.71, 160.0, 120.0


def get_joint_num(dataset):
    joint_num_dict = {'nyu': 14,'nyu_full': 23, 'icvl': 16, 'msra': 21, 'hands17': 21, 'itop': 15}
    return joint_num_dict[dataset]


def pixel2world(x, dataset):
    fx,fy,ux,uy = get_param(dataset)
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, dataset):
    fx,fy,ux,uy = get_param(dataset)
    x[:, :, 0] = x[:, :, 0] * fx/x[:, :, 2] + ux
    x[:, :, 1] = uy - x[:, :, 1] * fy / x[:, :, 2]
    return x


def jointImgTo3D(uvd, paras):
    fx, fy, fu, fv = paras
    ret = np.zeros_like(uvd, np.float32)
    if len(ret.shape) == 1:
        ret[0] = (uvd[0] - fu) * uvd[2] / fx
        ret[1] = (uvd[1] - fv) * uvd[2] / fy
        ret[2] = uvd[2]
    else:
        ret[:, 0] = (uvd[:,0] - fu) * uvd[:, 2] / fx
        ret[:, 1] = (uvd[:,1] - fv) * uvd[:, 2] / fy
        ret[:, 2] = uvd[:,2]
    return ret


def joint3DToImg(xyz, paras):
    fx, fy, fu, fv = paras
    ret = np.zeros_like(xyz, np.float32)
    if len(ret.shape) == 1:
        ret[0] = (xyz[0] * fx / xyz[2] + fu)
        ret[1] = (xyz[1] * fy / xyz[2] + fv)
        ret[2] = xyz[2]
    else:
        ret[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
        ret[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
        ret[:, 2] = xyz[:, 2]
    return ret


def save_result_img(index, root_dir,pic_dir, pose):
    img = cv2.imread(root_dir + '/convert/' + '{}.jpg'.format(index), 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    draw_pose(img, pose)
    cv2.imwrite(pic_dir+'/' + str(index) + ".png", img)


def get_sketch_setting(dataset):
    if dataset == 'icvl':
        return [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 9), (0, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15)]
    elif dataset == 'nyu':
        return [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (9, 10), (1, 13),
                (3, 13), (5, 13), (7, 13), (10, 13), (11, 13), (12, 13)]
    elif dataset == 'nyu_full':
        return [(20,3),(3,2),(2,1),(1,0),(20,7),(7,6),(6,5),(5,4),(20,11),(11,10),(10,9),(9,8),(20,15),(15,14),(14,13),(13,12),(20,19),(19,18),(18,17),(17,16),
               (20,21),(20,22)]
    elif dataset == 'msra':
        return [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]
    elif dataset == 'hands17':
        return [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 6), (6, 7), (7, 8),
                (2, 9), (9, 10), (10, 11), (3, 12), (12, 13), (13, 14), (4, 15), (15, 16),
                (16, 17), (5, 18), (18, 19), (19, 20)]
    elif dataset == 'itop':
        return [(0, 1),
                (1, 2), (2, 4), (4, 6),
                (1, 3), (3, 5), (5, 7),
                (1, 8),
                (8, 9), (9, 11), (11, 13),
                (8, 10), (10, 12), (12, 14)]


class Color(Enum):
    RED = (0, 0, 255)
    GREEN = (75, 255, 66)
    BLUE = (255, 0, 0)
    YELLOW = (204, 153, 17) #(17, 240, 244)
    PURPLE = (255, 255, 0)
    CYAN = (255, 0, 255)
    BROWN = (204, 153, 17)


def get_sketch_color(dataset):
    if dataset == 'icvl':
        return [Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'nyu':
        return (Color.GREEN, Color.RED, Color.PURPLE, Color.YELLOW, Color.BLUE, Color.BLUE, Color.GREEN,
                Color.RED, Color.PURPLE, Color.YELLOW, Color.BLUE, Color.CYAN, Color.CYAN)
    elif dataset == 'nyu_full':
        return (Color.GREEN,Color.GREEN,Color.GREEN,Color.GREEN, Color.RED, Color.RED, Color.RED, Color.RED,  Color.PURPLE, Color.PURPLE,Color.PURPLE,Color.PURPLE,
                Color.YELLOW,Color.YELLOW,Color.YELLOW,Color.YELLOW,
                Color.BLUE, Color.BLUE,  Color.BLUE, Color.BLUE,
                Color.CYAN, Color.CYAN)
    elif dataset == 'msra':
        return [Color.RED, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'hands17':
        return [Color.GREEN, Color.BLUE, Color.YELLOW, Color.PURPLE, Color.RED,
              Color.GREEN, Color.GREEN, Color.GREEN,
              Color.BLUE, Color.BLUE, Color.BLUE,
              Color.YELLOW, Color.YELLOW, Color.YELLOW,
              Color.PURPLE, Color.PURPLE, Color.PURPLE,
              Color.RED, Color.RED, Color.RED]
    elif dataset == 'itop':
        return [Color.RED,
              Color.GREEN, Color.GREEN, Color.GREEN,
              Color.BLUE, Color.BLUE, Color.BLUE,
                Color.CYAN,
              Color.YELLOW, Color.YELLOW, Color.YELLOW,
              Color.PURPLE, Color.PURPLE, Color.PURPLE,
              ]


def get_joint_color(dataset):
    if dataset == 'icvl':
        return [Color.CYAN, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'nyu':
        return (Color.GREEN, Color.GREEN, Color.RED, Color.RED, Color.PURPLE, Color.PURPLE, Color.YELLOW, Color.YELLOW,
                Color.BLUE, Color.BLUE, Color.BLUE,
                Color.CYAN, Color.CYAN, Color.CYAN)
    elif dataset == 'nyu_full':
        return (Color.GREEN, Color.GREEN,Color.GREEN, Color.GREEN, Color.RED, Color.RED, Color.RED, Color.RED,
                Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.BLUE, Color.BLUE, Color.BLUE,Color.BLUE,
                Color.CYAN, Color.CYAN, Color.CYAN)
    elif dataset == 'msra':
        return [Color.CYAN, Color.RED, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'hands17':
        return [Color.CYAN, Color.GREEN, Color.BLUE, Color.YELLOW, Color.PURPLE, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.PURPLE, Color.PURPLE, Color.PURPLE,
                Color.RED, Color.RED, Color.RED]
    elif dataset == 'itop':
        return  [Color.RED,Color.BROWN,
                 Color.GREEN, Color.BLUE, Color.GREEN, Color.BLUE, Color.GREEN, Color.BLUE,
                 Color.CYAN,
                 Color.YELLOW,Color.PURPLE,Color.YELLOW,Color.PURPLE,Color.YELLOW,Color.PURPLE]


def draw_pose(dataset, img, pose):
    colors = get_sketch_color(dataset)
    colors_joint = get_joint_color(dataset)
    idx = 0
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, colors_joint[idx].value, -1)
        idx = idx + 1
    idx = 0
    for x, y in get_sketch_setting(dataset):
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), colors[idx].value, 1)
        idx = idx + 1
    return img


def draw_depth_heatmap(dataset, pcl, heatmap, joint_id):
    fx, fy, ux, uy = get_param(dataset)
    pcl = pcl.transpose(1, 0)
    # pcl = joint3DToImg(pcl,(fx, fy, ux, uy))
    pcl = (pcl + 1) * 64
    sample_num = pcl.shape[0]
    img = np.ones((128, 128), dtype=np.uint8)*255
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors_joint = get_joint_color(dataset)
    for idx in range(sample_num):
        r = int(colors_joint[joint_id].value[0] * heatmap[joint_id,idx])
        b = int(colors_joint[joint_id].value[1] * heatmap[joint_id,idx])
        g = int(colors_joint[joint_id].value[2] * heatmap[joint_id,idx])
        cv2.circle(img, (int(pcl[idx,0]), int(pcl[idx,1])), 1,(r,g,b) , -1)
    return img


def debug_point_heatmap(dataset, data, index, GFM_):
    joint_num = len(get_joint_color(dataset))
    img, pcl_sample, joint_world, joint_img, center, M, cube, pcl_normal, joint_normal, offset, coeff, max_bbx_len = data
    img, joint_world, joint_img, pcl_sample = img.cuda(), joint_world.cuda(), joint_img.cuda(), pcl_sample.cuda()
    pcl_normal, joint_normal, offset, coeff, max_bbx_len = pcl_normal.cuda(), joint_normal.cuda(), offset.cuda(), coeff.cuda(), max_bbx_len.cuda()
    center, cube = center.cuda(),cube.cuda()
    feature = GFM_.joint2heatmap_pcl(joint_normal, pcl_normal, max_bbx_len)
    joint_predict = GFM_.heatmap2joint_pcl(feature, pcl_normal, max_bbx_len)
    joint_predict = torch.matmul((joint_predict + offset.view(-1, 1, 3).repeat(1, joint_num, 1)) * max_bbx_len.view(-1,1,1), coeff.inverse())
    joint_predict = (joint_predict - center.view(-1, 1, 3).repeat(1, joint_num, 1)) / cube.view(-1, 1, 3).repeat(1,joint_num,1) * 2
    print((joint_predict-joint_world).sum())
    for idx in range(pcl_normal.size(0)):
        for idx_joint in range(joint_num):
            img = draw_depth_heatmap(dataset, pcl_normal.cpu().numpy()[idx], feature.cpu().numpy()[idx],idx_joint)
            img_name = './debug/pcl_heatmap_' + str(index) + '_' + str(idx)+'_' + str(idx_joint) + '.png'
            cv2.imwrite(img_name, img)


def debug_2d_heatmap(img, joint_img, index, GFM_, dir_name='./debug/heatmap'):
    heatmap2d = GFM_.joint2heatmap2d(joint_img, isFlip=False, heatmap_size=img.size(-1))
    depth = GFM_.depth2map(joint_img[:, :, 2], heatmap_size=img.size(-1))
    feature = heatmap2d * (depth + 1) / 2
    for i in range(img.size(0)):
        for joint_index in range(joint_img.size(1)):
            depth = img.numpy()[i]
            heatmap = feature.numpy()[i]
            img_heatmap = ((depth + 1) / 4 + heatmap[joint_index] / 2) * 255.0
            img_name_1 = dir_name + '/' + str(index) + '_' + str(i)+'_'+str(joint_index) + '.png'
            cv2.imwrite(img_name_1, np.transpose(img_heatmap, (1, 2, 0)))


def debug_offset(data, batch_index, GFM_):
    img, pcl_sample, joint_world, joint_img, center, M, cube, pcl_normal, joint_normal, offset, coeff, max_bbx_len = data
    img_size = 32
    batch_size,joint_num,_ = joint_world.size()
    offset = GFM_.joint2offset(joint_img, img, feature_size=img_size)
    unit = offset[:, 0:joint_num*3, :, :].numpy()
    for index in range(batch_size):
        fig, ax = plt.subplots()
        unit_plam = unit[index, 0:3, :, :]
        x = np.arange(0,img_size,1)
        y = np.arange(0,img_size,1)

        X, Y = np.meshgrid(x, y)
        Y = img_size - 1 - Y
        ax.quiver(X, Y, unit_plam[0, ...], unit_plam[1, ...])
        ax.axis([0, img_size, 0, img_size])
        plt.savefig('./debug/offset_' + str(batch_index) + '_' + str(index) + '.png')


def debug_point_feature(data, index, GFM_):
    img, pcl_sample, joint_world, joint_img, center, M, cube, pcl_normal, joint_normal, offset, coeff, max_bbx_len = data
    joint_pc = GFM_.joint2pc(joint_normal)
    img_size = 128
    img_pcl = GFM_.pcl2img(joint_pc, img_size).squeeze(1).unsqueeze(-1) * 255
    for i in range(img.size(0)):
        img = img_pcl.numpy()[i]
        img[img > 0] = 255 / 2.0
        img_name_1 = './debug/pcl_' + str(index) + '_' + str(i) + '.png'
        cv2.imwrite(img_name_1, img)
    print(index)


def debug_2d_pose(img, joint_img, index, dataset, save_dir, save_name):
    batch_size, _, _, input_size = img.size()
    for img_idx in range(img.size(0)):
        joint_uvd = (joint_img + 1) / 2 * input_size
        img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
        img_show = draw_pose(dataset, cv2.cvtColor(img_draw[img_idx, 0], cv2.COLOR_GRAY2RGB), joint_uvd[img_idx])
        cv2.imwrite(save_dir + str(index*batch_size + img_idx) +'_' + save_name + '.png', img_show)
        return img_show