# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import torch
import numpy as np
import math
import cv2
import os
from PIL import Image
from torch.utils.data import DataLoader
import sys
import scipy.io as sio
import random
sys.path.append('..')

joint_select = np.array([0, 1, 3, 5,
                         6, 7, 9, 11,
                         12, 13, 15, 17,
                         18, 19, 21, 23,
                         24, 25, 27, 28,
                         32, 30, 31])
calculate = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 21, 22, 20]

xrange = range


def jitter_point_cloud(data, sigma=0.02, clip=0.1):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, jittered point clouds
    """
    N, C = data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += data
    return jittered_data


def Matr(axis, theta):
    M = np.eye(4)
    if axis == 0:
        M[1, 1] = np.cos(theta)
        M[1, 2] = -np.sin(theta)
        M[2, 1] = np.sin(theta)
        M[2, 2] = np.cos(theta)
    elif axis == 1:
        M[0, 0] = np.cos(theta)
        M[0, 2] = -np.sin(theta)
        M[2, 0] = np.sin(theta)
        M[2, 2] = np.cos(theta)
    elif axis == 2:
        M[0, 0] = np.cos(theta)
        M[0, 1] = -np.sin(theta)
        M[1, 0] = np.sin(theta)
        M[1, 1] = np.cos(theta)
    else:
        M[axis - 3, 3] = theta
    return M


def pixel2world(x, y, z, paras):
    fx,fy,fu,fv = paras
    worldX = (x - fu) * z / fx
    worldY = (fv - y) * z / fy
    return worldX, worldY


def pixel2world_noflip(x, y, z, paras):
    fx,fy,fu,fv = paras
    worldX = (x - fu) * z / fx
    worldY = (y - fv) * z / fy
    return worldX, worldY


def world2pixel(x, y, z, paras):
    fx, fy, fu, fv = paras
    pixelX = x * fx / z + fu
    pixelY = fv - y * fy / z
    return pixelX, pixelY


def get_center_from_bbx(depth, bbx, upper=807, lower=171):
    centers = np.array([0.0, 0.0, 300.0])
    img = depth[bbx[1]:bbx[3], bbx[0]:bbx[2]]
    flag = np.logical_and(img <= upper, img >= lower)
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    xv, yv = np.meshgrid(x, y)
    centers[0] = np.mean(xv[flag])
    centers[1] = np.mean(yv[flag])
    centers[2] = np.mean(img[flag])
    if centers[2] <= 0:
        centers[2] = 300.0
    if not flag.any():
        centers[0] = 0
        centers[1] = 0
        centers[2] = 300.0
    centers[0] += bbx[0]
    centers[1] += bbx[1]
    return centers


def get_center_fast(img, upper=650, lower=100):
    centers = np.array([0.0, 0.0, 300.0])
    flag = np.logical_and(img <= upper, img >= lower)
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    xv, yv = np.meshgrid(x, y)
    centers[0] = np.mean(xv[flag])
    centers[1] = np.mean(yv[flag])
    centers[2] = np.mean(img[flag])
    if centers[2] <= 0:
        centers[2] = 300.0
    if not flag.any():
        centers[0] = 0
        centers[1] = 0
        centers[2] = 300.0
    return centers


def rotatePoint2D(p1, center, angle):
    """
    Rotate a point in 2D around center
    :param p1: point in 2D (u,v,d)
    :param center: 2D center of rotation
    :param angle: angle in deg
    :return: rotated point
    """
    alpha = angle * np.pi / 180.
    pp = p1.copy()
    pp[0:2] -= center[0:2]
    pr = np.zeros_like(pp)
    pr[0] = pp[0] * np.cos(alpha) - pp[1] * np.sin(alpha)
    pr[1] = pp[0] * np.sin(alpha) + pp[1] * np.cos(alpha)
    pr[2] = pp[2]
    ps = pr
    ps[0:2] += center[0:2]
    return ps


def rotatePoints2D(pts, center, angle):
    """
    Transform points in 2D coordinates
    :param pts: point coordinates
    :param center: 2D center of rotation
    :param angle: angle in deg
    :return: rotated points
    """
    ret = pts.copy()
    for i in range(pts.shape[0]):
        ret[i] = rotatePoint2D(pts[i], center, angle)
    return ret


def transformPoint2D(pt, M):
    """
    Transform point in 2D coordinates
    :param pt: point coordinates
    :param M: transformation matrix
    :return: transformed point
    """
    pt2 = np.dot(np.asarray(M).reshape((3, 3)), np.asarray([pt[0], pt[1], 1]))
    return np.asarray([pt2[0] / pt2[2], pt2[1] / pt2[2]])


def transformPoints2D(pts, M):
    """
    Transform points in 2D coordinates
    :param pts: point coordinates
    :param M: transformation matrix
    :return: transformed points
    """
    ret = pts.copy()
    for i in range(pts.shape[0]):
        ret[i, 0:2] = transformPoint2D(pts[i, 0:2], M)
    return ret


def batchtransformPoints2D(pts, M):
    """
    Transform points in 2D coordinates
    :param pts: point coordinates
    :param M: transformation matrix
    :return: transformed points
    """
    new = []
    for index in range(pts.shape[0]):
        new.append(transformPoints2D(pts[index], M[index]))
    return np.stack(new, axis=0)


def hands17_reader(img_path):
    img = Image.open(img_path)  # open image
    assert len(img.getbands()) == 1  # ensure depth image
    depth = np.asarray(img, np.float32)
    return depth


def nyu_reader(img_path):
    img = cv2.imread(img_path)
    depth = np.asarray(img[:, :, 0] + img[:, :, 1] * 256, dtype=np.float32)
    return depth


def icvl_reader(img_path):
    img = Image.open(img_path)  # open image
    assert len(img.getbands()) == 1  # ensure depth image
    depth = np.asarray(img, np.float32)
    return depth


def msra_reader(image_name, para):
    f = open(image_name, 'rb')
    data = np.fromfile(f, dtype=np.uint32)
    width, height, left, top, right, bottom = data[:6]
    depth = np.zeros((height, width), dtype=np.float32)
    f.seek(4*6)
    data = np.fromfile(f, dtype=np.float32)
    depth[top:bottom, left:right] = np.reshape(data, (bottom-top, right-left))
    depth_pcl = np.reshape(data, (bottom-top, right-left))
    #convert to world
    imgHeight, imgWidth = depth_pcl.shape
    hand_3d = np.zeros([3, imgHeight*imgWidth])
    d2Output_x = np.tile(np.arange(imgWidth), (imgHeight, 1)).reshape(imgHeight, imgWidth).astype('float64') + left
    d2Output_y = np.repeat(np.arange(imgHeight), imgWidth).reshape(imgHeight, imgWidth).astype('float64') + top
    hand_3d[0], hand_3d[1] = pixel2world(d2Output_x.reshape(-1), d2Output_y.reshape(-1), depth_pcl.reshape(-1),para)
    hand_3d[2] = depth_pcl.reshape(-1)
    valid = np.arange(0,imgWidth*imgHeight)
    valid = valid[(hand_3d[0, :] != 0)|(hand_3d[1, :] != 0)|(hand_3d[2, :] != 0)]
    handpoints = hand_3d[:, valid].transpose(1,0)

    return depth,handpoints


def msra14_reader(image_name, para):
    f = open(image_name, 'rb')
    data = np.fromfile(f, dtype=np.float32)
    depth = np.reshape(data, (240, 320))
    return depth


def HO3D_reader(depth_filename):
    """Read the depth image in dataset and decode it"""
    depth_img = cv2.imread(depth_filename)

    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
    dpt = dpt
    return dpt


class loader(Dataset):
    def __init__(self, root_dir, phase, img_size, center_type, dataset_name):
        self.rng = np.random.RandomState(23455)
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.phase = phase
        self.img_size = img_size
        self.center_type = center_type
        self.allJoints = False
        # create OBB
        # self.pca = PCA(n_components=3)
        self.sample_num = 1024

    # numpy
    def jointImgTo3D(self, uvd, paras=None, flip=None):
        if isinstance(paras, tuple):
            fx, fy, fu, fv = paras
        else:
            fx, fy, fu, fv = self.paras
        if flip == None:
            flip = self.flip
        ret = np.zeros_like(uvd, np.float32)
        if len(ret.shape) == 1:
            ret[0] = (uvd[0] - fu) * uvd[2] / fx
            ret[1] = flip * (uvd[1] - fv) * uvd[2] / fy
            ret[2] = uvd[2]
        elif len(ret.shape) == 2:
            ret[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
            ret[:, 1] = flip * (uvd[:, 1] - fv) * uvd[:, 2] / fy
            ret[:, 2] = uvd[:, 2]
        else:
            ret[:, :, 0] = (uvd[:, :, 0] - fu) * uvd[:, :, 2] / fx
            ret[:, :, 1] = flip * (uvd[:, :, 1] - fv) * uvd[:, :, 2] / fy
            ret[:, :, 2] = uvd[:, :, 2]

        return ret

    def joint3DToImg(self, xyz, paras=None, flip=None):
        if isinstance(paras, tuple):
            fx, fy, fu, fv = paras
        else:
            fx, fy, fu, fv = self.paras
        if flip==None:
            flip = self.flip
        ret = np.zeros_like(xyz, np.float32)
        if len(ret.shape) == 1:
            ret[0] = (xyz[0] * fx / xyz[2] + fu)
            ret[1] = (flip * xyz[1] * fy / xyz[2] + fv)
            ret[2] = xyz[2]
        elif len(ret.shape) == 2:
            ret[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
            ret[:, 1] = (flip * xyz[:, 1] * fy / xyz[:, 2] + fv)
            ret[:, 2] = xyz[:, 2]
        else:
            ret[:, :, 0] = (xyz[:, :, 0] * fx / xyz[:, :, 2] + fu)
            ret[:, :, 1] = (flip * xyz[:, :, 1] * fy / xyz[:, :, 2] + fv)
            ret[:, :, 2] = xyz[:, :, 2]
        return ret

    # tensor
    def pointsImgTo3D(self, point_uvd, flip=None):
        if flip == None:
            flip = self.flip
        point_xyz = torch.zeros_like(point_uvd).to(point_uvd.device)
        point_xyz[:, :, 0] = (point_uvd[:, :, 0] - self.paras[2]) * point_uvd[:, :, 2] / self.paras[0]
        point_xyz[:, :, 1] = flip * (point_uvd[:, :, 1] - self.paras[3]) * point_uvd[:, :, 2] / self.paras[1]
        point_xyz[:, :, 2] = point_uvd[:, :, 2]
        return point_xyz

    def points3DToImg(self, joint_xyz, flip=None):
        fx, fy, fu, fv = self.paras
        if flip == None:
            flip = self.flip
        joint_uvd = torch.zeros_like(joint_xyz).to(joint_xyz.device)
        joint_uvd[:, :, 0] = (joint_xyz[:, :, 0] * fx / (joint_xyz[:, :, 2]+1e-8) + fu)
        joint_uvd[:, :, 1] = (flip * joint_xyz[:, :, 1] * fy / (joint_xyz[:, :, 2]) + fv)
        joint_uvd[:, :, 2] = joint_xyz[:, :, 2]
        return joint_uvd

    # augment
    def comToBounds(self, com, size, paras):
        fx, fy, fu, fv = paras
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        xstart = int(np.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2] * fx + 0.5))
        xend = int(np.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2] * fx + 0.5))
        ystart = int(np.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2] * fy + 0.5))
        yend = int(np.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2] * fy + 0.5))
        return xstart, xend, ystart, yend, zstart, zend

    def comToTransform(self, com, size, dsize, paras):
        """
        Calculate affine transform from crop
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :return: affine transform
        """

        xstart, xend, ystart, yend, _, _ = self.comToBounds(com, size, paras)

        trans = np.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart

        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            scale = np.eye(3) * dsize[0] / float(wb)
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            scale = np.eye(3) * dsize[1] / float(hb)
            sz = (wb * dsize[1] / hb, dsize[1])
        scale[2, 2] = 1

        # ori
        # xstart = int(np.floor(dsize[0] / 2. - sz[1] / 2.))
        # ystart = int(np.floor(dsize[1] / 2. - sz[0] / 2.))

        # change by pengfeiren
        xstart = int(np.floor(dsize[0] / 2. - sz[0] / 2.))
        ystart = int(np.floor(dsize[1] / 2. - sz[1] / 2.))
        off = np.eye(3)
        off[0, 2] = xstart
        off[1, 2] = ystart

        return np.dot(off, np.dot(scale, trans))

    def recropHand(self, crop, M, Mnew, target_size, paras, background_value=0., nv_val=0., thresh_z=True, com=None,
                   size=(250, 250, 250)):

        flags = cv2.INTER_NEAREST

        warped = cv2.warpPerspective(crop, np.dot(M, Mnew), target_size, flags=flags,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=float(background_value))
        # warped[np.isclose(warped, nv_val)] = background_value # Outliers will appear on the edge
        warped[warped < nv_val] = background_value

        if thresh_z is True:
            assert com is not None
            _, _, _, _, zstart, zend = self.comToBounds(com, size, paras)
            msk1 = np.logical_and(warped < zstart, warped != 0)
            msk2 = np.logical_and(warped > zend, warped != 0)
            warped[msk1] = zstart
            warped[msk2] = 0.  # backface is at 0, it is set later

        return warped


    def moveCoM(self, dpt, cube, com, off, joints3D, M, paras, pad_value=0):
        """
        Adjust already cropped image such that a moving CoM normalization is simulated
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param off: offset to center of mass (dx,dy,dz) in 3D coordinates
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """

        # if offset is 0, nothing to do
        if np.allclose(off, 0.):
            return dpt, joints3D, com, M

        # add offset to com
        new_com = self.joint3DToImg(self.jointImgTo3D(com) + off)

        # check for 1/0.
        if not (np.allclose(com[2], 0.) or np.allclose(new_com[2], 0.)):
            # scale to original size
            Mnew = self.comToTransform(new_com, cube, dpt.shape,paras)
            new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, paras, background_value=pad_value,
                                      nv_val=np.min(dpt[dpt>0])-1, thresh_z=True, com=new_com, size=cube)
        else:
            Mnew = M
            new_dpt = dpt

        # adjust joint positions to new CoM
        new_joints3D = joints3D + self.jointImgTo3D(com) - self.jointImgTo3D(new_com)

        return new_dpt, new_joints3D, new_com, Mnew

    def rotateHand(self, dpt, cube, com, rot, joints3D, pad_value=0):
        """
        Rotate hand virtually in the image plane by a given angle
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param rot: rotation angle in deg
        :param joints3D: original joint coordinates, in 3D coordinates (x,y,z)
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, rotation angle in XXX
        """

        # if rot is 0, nothing to do
        if np.allclose(rot, 0.):
            return dpt, joints3D, rot

        rot = np.mod(rot, 360)

        M = cv2.getRotationMatrix2D((dpt.shape[1] // 2, dpt.shape[0] // 2), -rot, 1)

        flags = cv2.INTER_NEAREST

        new_dpt = cv2.warpAffine(dpt, M, (dpt.shape[1], dpt.shape[0]), flags=flags,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=pad_value)

        new_dpt[new_dpt < (np.min(dpt[dpt > 0])-1)] = 0

        com3D = self.jointImgTo3D(com)
        joint_2D = self.joint3DToImg(joints3D + com3D)
        data_2D = np.zeros_like(joint_2D)
        for k in xrange(data_2D.shape[0]):
            data_2D[k] = rotatePoint2D(joint_2D[k], com[0:2], rot)
        new_joints3D = (self.jointImgTo3D(data_2D) - com3D)

        return new_dpt, new_joints3D, rot

    def scaleHand(self, dpt, cube, com, sc, joints3D, M, paras, pad_value=0):
        """
        Virtually scale the hand by applying different cube
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param sc: scale factor for cube
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """

        # if scale is 1, nothing to do
        if np.allclose(sc, 1.):
            return dpt, joints3D, cube, M

        new_cube = [s * sc for s in cube]

        # check for 1/0.
        if not np.allclose(com[2], 0.):
            # scale to original size
            Mnew = self.comToTransform(com, new_cube, dpt.shape, paras)
            new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, paras, background_value=pad_value,
                                      nv_val=np.min(dpt[dpt>0])-1, thresh_z=True, com=com, size=cube)
        else:
            Mnew = M
            new_dpt = dpt

        new_joints3D = joints3D

        return new_dpt, new_joints3D, new_cube, Mnew

    def jointmoveCoM(self, dpt, cube, com, off, joints3D, M, paras, pad_value=0):
        """
        Adjust already cropped image such that a moving CoM normalization is simulated
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param off: offset to center of mass (dx,dy,dz) in 3D coordinates
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """

        # if offset is 0, nothing to do
        if np.allclose(off, 0.):
            return joints3D,com,M

        # add offset to com
        new_com = self.joint3DToImg(self.jointImgTo3D(com) + off.reshape(1, 3))

        batch_size = joints3D.shape[0]
        Mnew = []
        for index in range(batch_size):
            # check for 1/0.
            if not (np.allclose(com[index,2], 0.) or np.allclose(new_com[index, 2], 0.)):
                # scale to original size
                Mnew.append(self.comToTransform(new_com[index], cube, dpt.shape, paras))
            else:
                Mnew.append(M)

        new_joints3D = []
        for index in range(batch_size):
            # adjust joint positions to new CoM
            new_joints3D.append(joints3D[index] + self.jointImgTo3D(com[index]) - self.jointImgTo3D(new_com[index]))

        return np.stack(new_joints3D, axis=0), new_com, np.stack(Mnew, axis=0)

    def jointrotateHand(self, dpt, cube, com, rot, joints3D, pad_value=0):
        """
        Rotate hand virtually in the image plane by a given angle
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param rot: rotation angle in deg
        :param joints3D: original joint coordinates, in 3D coordinates (x,y,z)
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, rotation angle in XXX
        """

        # if rot is 0, nothing to do
        if np.allclose(rot, 0.):
            return joints3D,rot

        rot = np.mod(rot, 360)
        com3D = self.jointImgTo3D(com)

        batch_size = joints3D.shape[0]
        new_joints3D = []
        for index in range(batch_size):
            joint_2D = self.joint3DToImg(joints3D[index] + com3D[index])
            data_2D = np.zeros_like(joint_2D)
            for k in xrange(data_2D.shape[0]):
                data_2D[k] = rotatePoint2D(joint_2D[k], com[index, 0:2], rot)
            new_joints3D.append(self.jointImgTo3D(data_2D) - com3D[index])

        return np.stack(new_joints3D, axis=0), rot

    def jointscaleHand(self,dpt, cube, com, sc, joints3D, M, paras, pad_value=0):
        """
        Virtually scale the hand by applying different cube
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param sc: scale factor for cube
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """

        # if scale is 1, nothing to do
        if np.allclose(sc, 1.):
            return joints3D, cube, M

        new_cube = [s * sc for s in cube]

        batch_size = joints3D.shape[0]
        Mnew = []
        for index in range(batch_size):
            # check for 1/0.
            if not np.allclose(com[index, 2], 0.):
                # scale to original size
                Mnew.append(self.comToTransform(com[index], new_cube, dpt.shape, paras))

            else:
                Mnew.append(M)
        new_joints3D = joints3D

        return new_joints3D, new_cube, np.stack(Mnew, axis=0)

    def rand_augment(self, sigma_com=None, sigma_sc=None, rot_range=None):
        if sigma_com is None:
            sigma_com = 35.

        if sigma_sc is None:
            sigma_sc = 0.05

        if rot_range is None:
            rot_range = 180.

        # mode = self.rng.randint(0, len(self.aug_modes))
        # off = self.rng.randn(3) * sigma_com  # +-px/mm
        # rot = self.rng.uniform(-rot_range, rot_range)
        # sc = abs(1. + self.rng.randn() * sigma_sc)
        #
        # mode = np.random.randint(0, len(self.aug_modes))
        # off = np.random.randn(3) * sigma_com  # +-px/mm
        # rot = np.random.uniform(-rot_range, rot_range)
        # sc = abs(1. + np.random.randn() * sigma_sc)

        mode = random.randint(0, len(self.aug_modes)-1)
        off = np.array([random.uniform(-1, 1) for a in range(3)]) * sigma_com# +-px/mm
        rot = random.uniform(-rot_range, rot_range)
        sc = abs(1. + random.uniform(-1,1) * sigma_sc)
        return mode, off, rot, sc

    def augmentCrop(self, img, gt3Dcrop, com, cube, M, mode, off, rot, sc, paras, normZeroOne=False):
        """
        Commonly used function to augment hand poses
        :param img: image
        :param gt3Dcrop: 3D annotations
        :param com: center of mass in image coordinates (x,y,z)
        :param cube: cube
        :param aug_modes: augmentation modes
        :param hd: hand detector
        :param normZeroOne: normalization
        :param sigma_com: sigma of com noise
        :param sigma_sc: sigma of scale noise
        :param rot_range: rotation range in degrees
        :return: image, 3D annotations(unnormal), com(image coordinates), cube
        """
        assert len(img.shape) == 2
        assert isinstance(self.aug_modes, list)
        premax = img.max()
        if np.max(img) == 0:
            imgD = img
            new_joints3D = gt3Dcrop
        elif self.aug_modes[mode] == 'com':
            rot = 0.
            sc = 1.
            imgD, new_joints3D, com, M = self.moveCoM(img.astype('float32'), cube, com, off, gt3Dcrop, M,paras, pad_value=0)
        elif self.aug_modes[mode] == 'rot':
            off = np.zeros((3,))
            sc = 1.
            imgD, new_joints3D, rot = self.rotateHand(img.astype('float32'), cube, com, rot, gt3Dcrop, pad_value=0)
        elif self.aug_modes[mode] == 'sc':
            off = np.zeros((3,))
            rot = 0.
            imgD, new_joints3D, cube, M = self.scaleHand(img.astype('float32'), cube, com, sc, gt3Dcrop, M, paras, pad_value=0)
        elif self.aug_modes[mode] == 'none':
            off = np.zeros((3,))
            sc = 1.
            rot = 0.
            imgD = img
            new_joints3D = gt3Dcrop
        else:
            raise NotImplementedError()
        imgD = self.normalize_img(premax, imgD, com, cube)
        return imgD, None, new_joints3D, np.asarray(cube), com, M, rot

    def batchaugmentCrop(self, img, batch_gt3Dcrop, com, cube, M, mode, off, rot, sc, paras, normZeroOne=False):
        """
        Commonly used function to augment hand poses
        :param img: image
        :param gt3Dcrop: 3D annotations
        :param com: center of mass in image coordinates (x,y,z)
        :param cube: cube
        :param aug_modes: augmentation modes
        :param hd: hand detector
        :param normZeroOne: normalization
        :param sigma_com: sigma of com noise
        :param sigma_sc: sigma of scale noise
        :param rot_range: rotation range in degrees
        :return: image, 3D annotations(unnormal), com(image coordinates), cube
        """
        assert len(img.shape) == 2
        assert isinstance(self.aug_modes, list)
        premax = img.max()
        if np.max(img) == 0:
            imgD = img
        elif self.aug_modes[mode] == 'com':
            rot = 0.
            sc = 1.
            new_batch_gt3Dcrop, com, M = self.jointmoveCoM(img.astype('float32'), cube, com, off, batch_gt3Dcrop, M,paras, pad_value=0)
        elif self.aug_modes[mode] == 'rot':
            off = np.zeros((3,))
            sc = 1.
            new_batch_gt3Dcrop, rot = self.jointrotateHand(img.astype('float32'), cube, com, rot, batch_gt3Dcrop, pad_value=0)
        elif self.aug_modes[mode] == 'sc':
            off = np.zeros((3,))
            rot = 0.
            new_batch_gt3Dcrop, cube, M = self.jointscaleHand(img.astype('float32'), cube, com, sc, batch_gt3Dcrop, M, paras,pad_value=0)
        elif self.aug_modes[mode] == 'none':
            off = np.zeros((3,))
            sc = 1.
            rot = 0.
            new_batch_gt3Dcrop = batch_gt3Dcrop
        else:
            raise NotImplementedError()
        return new_batch_gt3Dcrop, np.asarray(cube), com, M, rot

    def normalize_img(self, premax, imgD, com, cube):
        imgD[imgD == premax] = com[2] + (cube[2] / 2.)
        imgD[imgD == 0] = com[2] + (cube[2] / 2.)
        imgD[imgD >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
        imgD[imgD <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
        imgD -= com[2]
        imgD /= (cube[2] / 2.)
        return imgD

    # use deep-pp's method
    def Crop_Image_deep_pp(self, depth, com, size, dsize, paras):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """

        # print com, self.importer.jointImgTo3D(com)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(self.dpt, cmap=matplotlib.cm.jet)

        if len(size) != 3 or len(dsize) != 2:
            raise ValueError("Size must be 3D and dsize 2D bounding box")

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size,paras)

        # crop patch from source
        cropped = self.getCrop(depth, xstart, xend, ystart, yend, zstart, zend)

        # resize to same size
        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], int(hb * dsize[0] / wb))
        else:
            sz = (int(wb * dsize[1] / hb), dsize[1])

        trans = np.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart

        if cropped.shape[0] > cropped.shape[1]:
            scale = np.eye(3) * sz[1] / float(cropped.shape[0])
        else:
            scale = np.eye(3) * sz[0] / float(cropped.shape[1])


        scale[2, 2] = 1

        # depth resize
        rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_NEAREST)

        ret = np.ones(dsize, np.float32) * 0  # use background as filler
        xstart = int(np.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(np.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        # print rz.shape, xstart, ystart
        off = np.eye(3)
        off[0, 2] = xstart
        off[1, 2] = ystart

        return ret, np.dot(off, np.dot(scale, trans))

    # use deep-pp's method
    def Crop_Image_deep_pp_nodepth(self, com, size, dsize, paras):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """

        # print com, self.importer.jointImgTo3D(com)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(self.dpt, cmap=matplotlib.cm.jet)

        if len(size) != 3 or len(dsize) != 2:
            raise ValueError("Size must be 3D and dsize 2D bounding box")

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size,paras)
        # resize to same size
        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], int(hb * dsize[0] / wb))
        else:
            sz = (int(wb * dsize[1] / hb), dsize[1])

        trans = np.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        if hb > wb:
            scale = np.eye(3) * sz[1] / float(hb)
        else:
            scale = np.eye(3) * sz[0] / float(wb)
        scale[2, 2] = 1

        off = np.eye(3)
        xstart = int(np.floor(dsize[0] / 2. - sz[1] / 2.))
        ystart = int(np.floor(dsize[1] / 2. - sz[0] / 2.))
        off[0, 2] = xstart
        off[1, 2] = ystart

        return np.dot(off, np.dot(scale, trans))

    # for get trans
    def Batch_Crop_Image_deep_pp(self, depth, com, size, dsize, paras):
        batch_trans=[]
        for index in range(com.shape[0]):
            trans = self.Crop_Image_deep_pp_nodepth(com[index], size, dsize, paras)
            batch_trans.append(trans)
        return np.stack(batch_trans, axis=0)

    def getCrop(self, depth, xstart, xend, ystart, yend, zstart, zend, thresh_z=True, background=0):
        """
        Crop patch from image
        :param depth: depth image to crop from
        :param xstart: start x
        :param xend: end x
        :param ystart: start y
        :param yend: end y
        :param zstart: start z
        :param zend: end z
        :param thresh_z: threshold z values
        :return: cropped image
        """
        if len(depth.shape) == 2:
            cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1])].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0),
                                        abs(yend) - min(yend, depth.shape[0])),
                                       (abs(xstart) - max(xstart, 0),
                                        abs(xend) - min(xend, depth.shape[1]))), mode='constant',
                             constant_values=background)
        elif len(depth.shape) == 3:
            cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1]),
                      :].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0),
                                        abs(yend) - min(yend, depth.shape[0])),
                                       (abs(xstart) - max(xstart, 0),
                                        abs(xend) - min(xend, depth.shape[1])),
                                       (0, 0)), mode='constant', constant_values=background)
        else:
            raise NotImplementedError()

        if thresh_z is True:
            msk1 = np.logical_and(cropped < zstart, cropped != 0)
            msk2 = np.logical_and(cropped > zend, cropped != 0)
            cropped[msk1] = zstart
            cropped[msk2] = 0.  # backface is at 0, it is set later
        return cropped
    # point cloud
    def pca_point(self, pcl, joint):
        point_num = pcl.shape[0]
        if point_num < 10:
            pcl = self.joint2pc(joint)
        self.pca.fit(pcl)
        coeff = self.pca.components_.T
        if coeff[1, 0] < 0:
            coeff[:, 0] = -coeff[:, 0]
        if coeff[2, 2] < 0:
            coeff[:, 2] = -coeff[:, 2]
        coeff[:, 1] = np.cross(coeff[:, 2], coeff[:, 0])
        points_rotation = np.dot(pcl, coeff)
        joint_rotation = np.dot(joint, coeff)

        index = np.arange(points_rotation.shape[0])
        if points_rotation.shape[0] < self.sample_num:
            tmp = math.floor(self.sample_num / points_rotation.shape[0])
            index_temp = index.repeat(tmp)
            index = np.append(index_temp,
                              np.random.choice(index, size=divmod(self.sample_num, points_rotation.shape[0])[1],
                                               replace=False))
        index = np.random.choice(index, size=self.sample_num, replace=False)
        points_rotation_sampled = points_rotation[index]

        # Normalize Point Cloud
        scale = 1.2
        bb3d_x_len = scale * (points_rotation[:, 0].max() - points_rotation[:, 0].min())
        bb3d_y_len = scale * (points_rotation[:, 1].max() - points_rotation[:, 1].min())
        bb3d_z_len = scale * (points_rotation[:, 2].max() - points_rotation[:, 2].min())
        max_bb3d_len = bb3d_x_len / 2.0

        points_rotation_sampled_normalized = points_rotation_sampled / max_bb3d_len
        joint_rotation_normalized = joint_rotation / max_bb3d_len
        if points_rotation.shape[0] < self.sample_num:
            offset = np.mean(points_rotation, 0) / max_bb3d_len
        else:
            offset = np.mean(points_rotation_sampled_normalized, 0)
        points_rotation_sampled_normalized = points_rotation_sampled_normalized - offset
        joint_rotation_normalized = joint_rotation_normalized - offset
        return points_rotation_sampled_normalized, joint_rotation_normalized, offset, coeff, max_bb3d_len

    def joint2pc(self, joint, radius=15):
        joint_num, _ = joint.shape

        radius = np.random.rand(joint_num, 100) * radius
        theta = np.random.rand(joint_num, 100) * np.pi
        phi = np.random.rand(joint_num, 100) * np.pi

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        point = np.tile(joint[:, np.newaxis, :], (1, 100, 1)) + np.concatenate(
            (x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis=-1)
        point = point.reshape([100 * joint_num, 3])
        sample = np.random.choice(100 * joint_num, self.sample_num, replace=False)
        return point[sample, :]

    #return normalied pcl
    def getpcl(self, imgD, com3D, cube, M):
        mask = np.where(imgD > 0.99)
        dpt_ori = imgD * cube[2] / 2.0 + com3D[2]
        # change the background value to 1
        dpt_ori[mask] = 0

        pcl = (self.depthToPCL(dpt_ori, M) - com3D)
        pcl_num = pcl.shape[0]
        cube_tile = np.tile(cube / 2.0, pcl_num).reshape([pcl_num, 3])
        pcl = pcl / cube_tile
        return pcl


    def farthest_point_sample(self, xyz, npoint):
        N, C = xyz.shape
        S = npoint
        if N < S:
            centroids = np.arange(N)
            centroids = np.append(centroids, np.random.choice(centroids, size=S - N, replace=False))
        else:
            centroids = np.zeros(S).astype(np.int)
            distance = np.ones(N) * 1e10
            farthest = np.random.randint(0, S)
            for i in range(S):
                centroids[i] = farthest
                centroid = xyz[farthest, :]
                dist = np.sum((xyz - centroid) ** 2, -1)
                mask = dist < distance
                distance[mask] = dist[mask]
                farthest = distance.argmax()
        return np.unique(centroids)

    def depthToPCL(self, dpt, T, background_val=0.):
        fx, fy, fu, fv = self.paras
        # get valid points and transform
        pts = np.asarray(np.where(~np.isclose(dpt, background_val))).transpose()
        pts = np.concatenate([pts[:, [1, 0]] + 0.5, np.ones((pts.shape[0], 1), dtype='float32')], axis=1)
        pts = np.dot(np.linalg.inv(np.asarray(T)), pts.T).T
        pts = (pts[:, 0:2] / pts[:, 2][:, None]).reshape((pts.shape[0], 2))

        # replace the invalid data
        depth = dpt[(~np.isclose(dpt, background_val))]

        # get x and y data in a vectorized way
        row = (pts[:, 0] - fu) / fx * depth
        col = self.flip * (pts[:, 1] - fv) / fy * depth

        # combine x,y,depth
        return np.column_stack((row, col, depth))

    # tensor
    def unnormal_joint_img(self, joint_img):
        device = joint_img.device
        joint = torch.zeros(joint_img.size()).to(device)
        joint[:, :, 0:2] = (joint_img[:, :, 0:2] + 1) / 2 * self.img_size
        joint[:, :, 2] = (joint_img[:, :, 2] + 1) / 2 * self.cube_size[2]
        return joint

    def uvd_nl2xyznl_tensor(self, uvd, center, m, cube):
        batch_size, point_num, _ = uvd.size()
        device = uvd.device
        cube_size_t = cube.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
        M_t = m.to(device).view(batch_size, 1, 3, 3).repeat(1, point_num, 1, 1)
        M_inverse = torch.inverse(M_t)

        uv_unnormal= (uvd[:, :, 0:2] + 1) * (self.img_size / 2)
        d_unnormal = (uvd[:, :, 2:]) * (cube_size_t[:, :, 2:] / 2.0) + center_t[:, :, 2:]
        uvd_unnormal = torch.cat((uv_unnormal,d_unnormal),dim=-1)
        uvd_world = self.get_trans_points(uvd_unnormal, M_inverse)
        xyz = self.pointsImgTo3D(uvd_world)
        xyz_noraml = (xyz - center_t) / (cube_size_t / 2.0)
        return xyz_noraml

    def xyz_nl2uvdnl_tensor(self, joint_xyz, center, M, cube_size):
        device = joint_xyz.device
        batch_size, joint_num, _ = joint_xyz.size()
        cube_size_t = cube_size.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        M_t = M.to(device).view(batch_size, 1, 3, 3).repeat(1, joint_num, 1, 1)

        joint_temp = joint_xyz * cube_size_t / 2.0 + center_t
        joint_uvd = self.points3DToImg(joint_temp)
        joint_uvd = self.get_trans_points(joint_uvd, M_t)
        joint_uv = joint_uvd[:, :, 0:2] / self.img_size * 2.0 - 1
        joint_d = (joint_uvd[:, :, 2:] - center_t[:, :, 2:]) / (cube_size_t[:, :, 2:] / 2)
        joint = torch.cat((joint_uv, joint_d), dim=-1)
        return joint

    # get point feature from 2d feature
    def imgFeature2pclFeature(self, pcl_uvd, feature):
        '''
        :param pcl: BxN*3 Tensor
        :param feature: BxCxWxH Tensor
        :param center:  Tensor
        :param M:  FloatTensor
        :param cube_size:  LongTensor
        :return: select_feature: BxCxN
        '''

        batch_size, point_num, _ = pcl_uvd.size()
        feature_size = feature.size(-1)
        feature_dim = feature.size(1)

        pcl_uvd = torch.clamp(pcl_uvd,-1,1)
        pcl_uvd = (pcl_uvd+1)/2.0 * (feature_size)
        uv_idx = torch.floor(pcl_uvd[:,:,1]) * feature_size + torch.floor(pcl_uvd[:,:,0])
        uv_idx = uv_idx.long().view(batch_size,1,point_num).repeat(1,feature_dim,1)
        select_feature = torch.gather(feature.view(batch_size,-1, feature_size*feature_size), dim=-1, index=uv_idx).view(batch_size,feature_dim,point_num)

        return select_feature

    def get_trans_points(self, joints, M):
        device = joints.device
        joints_mat = torch.cat((joints[:, :, 0:2], torch.ones(joints.size(0), joints.size(1), 1).to(device)), dim=-1)
        joints_trans_xy = torch.matmul(M, joints_mat.unsqueeze(-1)).squeeze(-1)[:, :, 0:2]
        joints_trans_z = joints[:, :, 2:]
        return torch.cat((joints_trans_xy,joints_trans_z),dim=-1)

    # return the same size tensor as img, which stand for uvd coord.
    def Img2uvd(self, img, feature_size, center, M, cube):
        batch_size = img.size(0)
        device = img.device
        img_rs = F.interpolate(img,(feature_size,feature_size))
        mask = img_rs.ge(0.99)
        mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_y,mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, img_rs), dim=1)
        return coords
        # pcl = torch.cat((coords, img_rs), dim=1).view(batch_size, 3, feature_size*feature_size).permute(0, 2, 1)
        # pcl = self.uvd_nl2xyznl_tensor(pcl, center, M, cube)
        # return pcl.view(batch_size, 3, feature_size, feature_size)

    def xyzImg2uvdImg(self, xyz_img,render_size, center, m, cube):
        batch_size = xyz_img.size()[0]
        feature_size = xyz_img.size()[-1]
        device = xyz_img.device
        mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_y,mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, xyz_img), dim=1)
        uvdPoint = self.xyz_nl2uvdnl_tensor(coords.view(batch_size,3,feature_size*feature_size).permute(0,2,1), center, m, cube)
        # pcl = torch.cat((coords,img_rs), dim=1).view(batch_size,3,feature_size*feature_size).permute(0,2,1)
        # pcl = self.uvd_nl2xyznl_tensor(pcl, center, m, cube)
        # return pcl.view(batch_size, 3, feature_size, feature_size)
        uvd_img = self.drawpcl(uvdPoint, render_size)
        return uvd_img

    def render_depth(self, mesh, center_xyz, rot=(0.0, 0.0, 0.0), cube_size=250, scale=0.5):
        img_width, img_height = int(scale*640), int(scale*480)
        node_num, _ = mesh.shape

        # # rotate in world coord
        # mesh_xyz = mesh - center_xyz
        # mesh_xyz[:, 2] = - mesh_xyz[:, 2]
        # mesh_xyz = np.concatenate((mesh_xyz, np.ones([node_num,1])), axis=-1)
        # mesh_xyz = np.dot(mesh_xyz, Matr(2, rot[2]).T)
        # mesh_xyz = np.dot(mesh_xyz, Matr(0, rot[0]).T)
        # mesh_xyz = np.dot(mesh_xyz, Matr(1, rot[1]).T)
        # mesh_uvd = self.joint3DToImg(mesh_xyz[:, 0:3] + center_xyz)
        # mesh_xyz = mesh - center_xyz
        # mesh_xyz[:, 2] = - mesh_xyz[:, 2]
        # mesh_xyz = mesh_xyz + center_xyz
        mesh_uvd = self.joint3DToImg(mesh)

        # normal depth
        mesh_uvd[:, 2] = (mesh_uvd[:, 2] - center_xyz[2]) / cube_size * 2
        img = np.ones([img_width * img_height]) * -1

        # mesh_grid_x = torch.clamp(torch.floor((mesh[:, :, 0] + 1) / 2 * img_size), 0, img_size - 1)
        # mesh_grid_y = torch.clamp(torch.floor((-mesh[:, :, 1] + 1) / 2 * img_size), 0, img_size - 1)
        # mesh_grid_z = (mesh[:, :, 2] + 1) / 2
        # value = mesh_grid_x + mesh_grid_y*img_size + mesh_grid_z
        # value_sort, indices = torch.sort(value)
        # value_int = torch.floor(value_sort)
        # # because gpu can't assign value in order
        # img[:, value_int.long()] = (value_sort - value_int).cpu()

        mesh_grid_x = np.ceil(np.clip(mesh_uvd[:, 1] * scale, 0, img_height - 1))
        mesh_grid_y = np.ceil(np.clip(mesh_uvd[:, 0] * scale, 0, img_width - 1))
        mesh_grid_z = np.clip((mesh_uvd[:, 2] + 1) / 2, 0, 1)  # resize to [0,1]
        value = mesh_grid_x * img_width + mesh_grid_y + mesh_grid_z

        value_sort = np.sort(value)
        value_sort = value_sort[::-1]
        value_int = np.floor(value_sort).astype('int64')
        img[value_int] = ((value_sort - value_int) * 2 - 1)# recover to [-1,1]
        mask = np.ones_like(img)
        mask[img < -0.99] = 0
        img = mask * (img * cube_size / 2.0 + center_xyz[2])
        img = img.reshape(img_height, img_width)
        img = cv2.resize(img, (640, 480))
        return img

    # mesh world_coord
    # weight sum to 1
    def weight_pcl2depht(self, mesh, weight):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import matplotlib.colors as colors

        cNorm = colors.Normalize(vmin=0, vmax=1.0)
        jet = plt.get_cmap('jet')
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
        weight = weight/weight.max()
        img = np.zeros([480*640])
        mesh_uv = self.joint3DToImg(mesh)[:,0:2].astype(np.int)
        index = (mesh_uv[:,1] * 640 + mesh_uv[:,0])
        img[index] = weight
        img = img.reshape((480,640))
        img_color = 255 * scalarMap.to_rgba(1- img)
        return img_color

    def read_modelPara(self, data_rt, view):
        theta = np.loadtxt(data_rt+'/posePara_lm_collosion/'+self.dataset_name+'-'+self.phase+'-'+str(view)+'-pose.txt').reshape(-1, 45)
        quat = np.loadtxt(data_rt+'/posePara_lm_collosion/'+self.dataset_name+'-'+self.phase+'-'+str(view)+'-glb.txt').reshape(-1, 3)
        scale = np.loadtxt(data_rt+'/posePara_lm_collosion/'+self.dataset_name+'-'+self.phase+'-'+str(view)+'-scale.txt').reshape(-1, 1)
        trans = np.loadtxt(data_rt+'/posePara_lm_collosion/'+self.dataset_name+'-'+self.phase+'-'+str(view)+'-trans.txt').reshape(-1, 3)
        shape = np.loadtxt(data_rt+'/posePara_lm_collosion/'+self.dataset_name+'-'+self.phase+'-'+str(view)+'-shape.txt').reshape(-1, 10)

        model_para = np.concatenate([quat, theta, shape, scale, trans], axis=-1)
        return model_para
    # 计算关节点可视性
    def calculate_visible(self, pcl, joint):
        dis_value = 20
        # dis_value = np.array([[15], [20], [15], [20], [15], [20], [15], [20], [20], [25], [25], [30], [30], [30]])
        # dis_value = np.array([[15], [20], [15], [20], [15], [20], [15], [20], [20], [25], [25], [30], [30], [30]])
        offset = pcl.reshape(1, -1, 3) - joint.reshape(-1, 1, 3)
        dis = np.sqrt(np.sum(offset * offset, axis=-1))
        # dis_min = np.amin(dis, axis=-1)
        visible = (dis < dis_value).sum(axis=-1) > 20
        return visible
        # dis_min_idx = np.argmin(dis, axis=-1)
        # idx_num = np.bincount(dis_min_idx, minlength=mesh.shape[0])
        # idx_binary = idx_num > 0
        # weight_sum = np.sum(self.mesh_weight[:, idx_binary], axis=-1)
        # visible = weight_sum < 0.25


class double_memory_nyu_loader(loader):
    def __init__(self, root_dir, phase, percent=1.0, view=0, aug_para=[10, 0.1, 180], img_size=128, cube_size=[300,300,300], center_type='refine', joint_num=14, loader=nyu_reader):
        super(double_memory_nyu_loader, self).__init__(root_dir, phase, img_size, center_type, 'nyu')
        np.random.seed(1)
        self.paras = (588.03, 587.07, 320., 240.)
        self.cube_size = np.array(cube_size)
        self.allJoints = True
        self.flip = -1
        if phase == 'test':
            self.percent = 1.0
        else:
            self.percent = percent

        self.croppedSz = img_size
        self.joint_num = joint_num
        self.aug_modes = ['rot', 'com', 'sc','none']#'rot','com','sc','none'
        self.aug_para = aug_para

        data_path = '{}/{}'.format(self.root_dir, self.phase)
        label_path = '{}/joint_data.mat'.format(data_path)
        center_path ='{}/center_{}_refined.txt'.format(data_path, self.phase)
        print('loading data...')
        self.labels = sio.loadmat(label_path)
        self.data_path = data_path
        self.view = view

        self.all_joints_uvd = self.labels['joint_uvd'][self.view][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.labels['joint_xyz'][self.view][:, joint_select, :][:, calculate, :]
        self.refine_center_xyz = np.loadtxt(center_path)
        self.selected_idx = np.random.choice(np.arange(len(self.all_joints_uvd)), int(len(self.all_joints_uvd)*self.percent),replace=False)
        self.mask = np.zeros([len(self.all_joints_uvd)])
        self.mask[self.selected_idx] = 1

        print('finish!!')
        if center_type =='refine':
            self.center_xyz = self.refine_center_xyz
        elif center_type =='joint':
            self.center_xyz = self.all_joints_xyz[:,20,:]
        elif center_type =='joint_mean':
            self.center_xyz = self.all_joints_xyz.mean(1)
        elif center_type == 'random':
            self.center_xyz = self.all_joints_xyz.mean(1)

        self.loader = loader
        self.test_cubesize = np.ones([8252, 3])*self.cube_size
        self.test_cubesize[2440:, :] = self.test_cubesize[2440:, :] * 5.0 / 6.0
        if joint_num == 14:
            self.allJoints = False
        else:
            self.allJoints = True

    def __getitem__(self, index):
        img_path = self.data_path + '/depth_'+str(self.view+1)+'_{:07d}.png'.format(index+1)
        depth = self.loader(img_path)
        joint_xyz = self.all_joints_xyz[index].copy()
        mask = self.mask[index]

        if self.phase == 'test':
            cube_size = self.test_cubesize[index]
        else:
            cube_size = self.cube_size

        if self.center_type == 'random':
            random_trans = (np.random.rand(3) - 0.5) * 2 * 0.2 * cube_size
            center_xyz = self.center_xyz[index] + random_trans
            center_uvd = self.joint3DToImg(center_xyz)
        elif self.center_type == 'mean':
            center_xyz = joint_xyz.mean(0)
            center_uvd = self.joint3DToImg(center_xyz)
        else:
            center_xyz = self.center_xyz[index]
            center_uvd = self.joint3DToImg(center_xyz)

        gt3Dcrop = joint_xyz - center_xyz
        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size,self.img_size), self.paras)
        if self.phase == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
            imgD, _, curLabel,  cube, com2D, M, _ = self.augmentCrop(depth_crop.copy(), gt3Dcrop.copy(), center_uvd.copy(), self.cube_size, trans.copy(), mode, off, rot, sc, self.paras)
            curLabel = curLabel / (cube[2] / 2.0)

            mode_tf, off_tf, rot_tf, sc_tf = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
            imgD_tf, _, curLabel_tf,  cube_tf, com2D_tf, M_tf, _ = self.augmentCrop(depth_crop.copy(), gt3Dcrop.copy(), center_uvd.copy(), self.cube_size, trans.copy(), mode_tf, off_tf, rot_tf, sc_tf, self.paras)
            curLabel_tf = curLabel_tf / (cube_tf[2] / 2.0)
        else:
            imgD = self.normalize_img(depth_crop.max(),depth_crop, center_xyz, cube_size)
            # synth_imgD = self.normalize_img(synth_depth_crop.max(), synth_depth_crop, center_xyz, cube_size)
            curLabel = gt3Dcrop / (cube_size[2] / 2.0)
            cube = np.array(cube_size)
            com2D = center_uvd
            M = trans
            imgD_tf = imgD
            curLabel_tf = curLabel
            cube_tf = cube
            com2D_tf = com2D

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)

        com3D_tf = self.jointImgTo3D(com2D_tf)
        joint_img_tf = transformPoints2D(self.joint3DToImg(curLabel_tf * (cube_tf[0] / 2.0) + com3D_tf), M_tf)
        joint_img_tf[:, 0:2] = joint_img_tf[:, 0:2] / (self.img_size / 2) - 1
        joint_img_tf[:, 2] = (joint_img_tf[:, 2] - com3D_tf[2]) / (cube[0] / 2.0)

        # get pcl
        pcl = self.getpcl(imgD, com3D, cube, M)
        pcl_index = np.arange(pcl.shape[0])
        pcl_num = pcl.shape[0]
        if pcl_num == 0:
            pcl_sample = np.zeros([self.sample_num, 3])
        else:
            if pcl_num < self.sample_num:
                tmp = math.floor(self.sample_num / pcl_num)
                index_temp = pcl_index.repeat(tmp)
                pcl_index = np.append(index_temp, np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1], replace=False))
            select = np.random.choice(pcl_index, self.sample_num, replace=False)
            pcl_sample = pcl[select, :]
        pcl_sample = torch.from_numpy(pcl_sample).float()

        data = torch.from_numpy(imgD).float().unsqueeze(0)
        joint_img = torch.from_numpy(joint_img).float()*mask
        joint = torch.from_numpy(curLabel).float()

        data_tf = torch.from_numpy(imgD_tf).float().unsqueeze(0)
        joint_img_tf = torch.from_numpy(joint_img_tf).float()*mask
        joint_tf = torch.from_numpy(curLabel_tf).float()

        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()
        so_node = torch.ones([self.joint_num, 3])
        visible = torch.ones([1])*mask
        outline = torch.ones([1])*mask

        return data, data_tf, pcl_sample, joint, joint_tf, joint_img, joint_img_tf, so_node, center, M, cube, visible, outline

    def __len__(self):
        return len(self.all_joints_xyz)


class memory_nyu_loader(loader):
    def __init__(self, root_dir, phase, percent=1.0, view=0, aug_para=[10, 0.1, 180], img_size=128, cube_size=[300,300,300], center_type='refine', joint_num=14, loader=nyu_reader):
        super(memory_nyu_loader, self).__init__(root_dir, phase, img_size, center_type, 'nyu')
        np.random.seed(1)
        self.paras = (588.03, 587.07, 320., 240.)
        self.cube_size = np.array(cube_size)
        self.allJoints = True
        self.flip = -1
        if phase == 'test':
            self.percent = 1.0
        else:
            self.percent = percent

        self.croppedSz = img_size
        self.joint_num = joint_num
        self.aug_modes = ['rot', 'com', 'sc','none']#'rot','com','sc','none'
        self.aug_para = aug_para

        data_path = '{}/{}'.format(self.root_dir, self.phase)
        label_path = '{}/joint_data.mat'.format(data_path)
        center_path ='{}/center_{}_refined.txt'.format(data_path, self.phase)
        print('loading data...')
        self.labels = sio.loadmat(label_path)
        self.data_path = data_path
        self.view = view

        self.all_joints_uvd = self.labels['joint_uvd'][self.view][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.labels['joint_xyz'][self.view][:, joint_select, :][:, calculate, :]
        self.refine_center_xyz = np.loadtxt(center_path)
        self.selected_idx = np.random.choice(np.arange(len(self.all_joints_uvd)), int(len(self.all_joints_uvd)*self.percent),replace=False)
        self.mask = np.zeros([len(self.all_joints_uvd)])
        self.mask[self.selected_idx] = 1

        print('finish!!')
        if center_type =='refine':
            self.center_xyz = self.refine_center_xyz
        elif center_type =='joint':
            self.center_xyz = self.all_joints_xyz[:,20,:]
        elif center_type =='joint_mean':
            self.center_xyz = self.all_joints_xyz.mean(1)
        elif center_type == 'random':
            self.center_xyz = self.all_joints_xyz.mean(1)

        self.loader = loader
        self.test_cubesize = np.ones([8252, 3])*self.cube_size
        self.test_cubesize[2440:, :] = self.test_cubesize[2440:, :] * 5.0 / 6.0
        if joint_num == 14:
            self.allJoints = False
        else:
            self.allJoints = True

    def __getitem__(self, index):
        img_path = self.data_path + '/depth_'+str(self.view+1)+'_{:07d}.png'.format(index+1)
        depth = self.loader(img_path)
        joint_xyz = self.all_joints_xyz[index].copy()
        mask = self.mask[index]

        if self.phase == 'test':
            cube_size = self.test_cubesize[index]
        else:
            cube_size = self.cube_size

        if self.center_type == 'random':
            random_trans = (np.random.rand(3) - 0.5) * 2 * 0.2 * cube_size
            center_xyz = self.center_xyz[index] + random_trans
            center_uvd = self.joint3DToImg(center_xyz)
        elif self.center_type == 'mean':
            center_xyz = joint_xyz.mean(0)
            center_uvd = self.joint3DToImg(center_xyz)
        else:
            center_xyz = self.center_xyz[index]
            center_uvd = self.joint3DToImg(center_xyz)

        gt3Dcrop = joint_xyz - center_xyz
        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size,self.img_size), self.paras)
        # synth_depth_crop, trans = self.Crop_Image_deep_pp(synth_depth, center_uvd, cube_size, dsize=(self.img_size,self.img_size))

        if self.phase == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
            imgD, _, curLabel,  cube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, self.cube_size,
                                                                    trans, mode, off, rot, sc, self.paras)
            # synth_imgD, _, mesh_curLabel, cube, com2D, M, _ = self.augmentCrop(synth_depth_crop, gt3Dcrop, center_uvd, self.cube_size,
            #                                                         trans, mode, off, rot, sc)
            curLabel = curLabel / (cube[2] / 2.0)
        else:
            imgD = self.normalize_img(depth_crop.max(),depth_crop, center_xyz, cube_size)
            # synth_imgD = self.normalize_img(synth_depth_crop.max(), synth_depth_crop, center_xyz, cube_size)
            curLabel = gt3Dcrop / (cube_size[2] / 2.0)
            cube = np.array(cube_size)
            com2D = center_uvd
            M = trans

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)

        # get pcl
        pcl = self.getpcl(imgD, com3D, cube, M)
        pcl_index = np.arange(pcl.shape[0])
        pcl_num = pcl.shape[0]
        if pcl_num == 0:
            pcl_sample = np.zeros([self.sample_num, 3])
        else:
            if pcl_num < self.sample_num:
                tmp = math.floor(self.sample_num / pcl_num)
                index_temp = pcl_index.repeat(tmp)
                pcl_index = np.append(index_temp, np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1], replace=False))
            select = np.random.choice(pcl_index, self.sample_num, replace=False)
            pcl_sample = pcl[select, :]
        pcl_sample = torch.from_numpy(pcl_sample).float()

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        joint_img = torch.from_numpy(joint_img).float()*mask
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()
        so_node = torch.ones([self.joint_num, 3])
        visible = torch.ones([1])*mask
        outline = torch.ones([1])*mask

        return data, pcl_sample, joint, joint_img, so_node, center, M, cube, visible, outline

    def __len__(self):
        return len(self.all_joints_xyz)


def mask_joints(img, img_joint, para, min_mask_num=3, max_mask_num=10):
    mask_num = np.random.choice(np.arange(min_mask_num, max_mask_num), 1,replace=False)[0]
    img_size = img.shape[-1]
    joint_num = img_joint.shape[0]
    joint_id = np.random.choice(np.arange(0, joint_num), mask_num, replace=False)
    mask_uvd = img_joint[joint_id, :]
    uvd_offset = (np.random.random(mask_uvd.shape)-0.5)*para*2
    mask_uvd = mask_uvd+uvd_offset
    mask_range = (np.random.uniform(0, 1, mask_num) + 1) * para
    xx, yy = np.meshgrid(np.arange(img_size), np.arange(img_size))
    xx = 2*(xx + 0.5) / img_size - 1.0
    yy = 2*(yy + 0.5) / img_size - 1.0
    mesh = np.stack((xx, yy, img), axis=-1).reshape([1, -1, 3])
    dis = np.sqrt(np.sum((mesh - mask_uvd.reshape([mask_num, 1, 3]))**2, axis=-1))
    mask = dis < mask_range.reshape([mask_num,1])
    mask = 1 - mask.sum(0)>0
    return np.where(mask.reshape([128, 128]), img, np.ones_like(img))


class nyu_loader(loader):
    def __init__(self, root_dir, phase, aug_para=[10, 0.1, 180], img_size=128,
                 cube_size=[250, 250, 250], center_type='refine', joint_num=14, loader=nyu_reader):
        super(nyu_loader, self).__init__(root_dir, phase, img_size, center_type, 'nyu')
        self.paras = (588.03, 587.07, 320., 240.)
        self.cube_size = np.array(cube_size)
        self.allJoints = True
        self.flip = -1

        self.croppedSz = img_size
        self.joint_num = joint_num
        self.aug_modes = ['rot', 'com', 'sc','none']#'rot','com','sc','none'
        self.aug_para = aug_para

        data_path = '{}/{}'.format(self.root_dir, self.phase)
        label_path = '{}/joint_data.mat'.format(data_path)
        center_path ='{}/center_{}_refined.txt'.format(data_path, self.phase)
        print('loading data...')
        self.labels = sio.loadmat(label_path)
        self.data_path = data_path

        self.all_joints_uvd = self.labels['joint_uvd'][0][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.labels['joint_xyz'][0][:, joint_select, :][:, calculate, :]
        self.refine_center_xyz = np.loadtxt(center_path)

        print('finish!!')
        if center_type =='refine':
            self.center_xyz = self.refine_center_xyz
        elif center_type =='joint':
            self.center_xyz = self.all_joints_xyz[:,20,:]
        elif center_type =='joint_mean':
            self.center_xyz = self.all_joints_xyz.mean(1)
        elif center_type == 'random':
            self.center_xyz = self.all_joints_xyz.mean(1)

        self.loader = loader
        self.test_cubesize = np.ones([8252, 3])*self.cube_size
        self.test_cubesize[2440:, :] = self.test_cubesize[2440:, :] * 5.0 / 6.0
        if joint_num == 14:
            self.allJoints = False
        else:
            self.allJoints = True

    def __getitem__(self, index):
        if self.phase == 'train':
            img_path = self.data_path + '/depth_1_{:07d}.png'.format(index + 1)
        else:
            img_path = self.data_path + '/depth_1_{:07d}.png'.format(index + 1)
        if not os.path.exists(img_path):
            print(img_path)
        depth = self.loader(img_path)
        joint_xyz = self.all_joints_xyz[index].copy()
        if self.phase == 'test':
            cube_size = self.test_cubesize[index]
        else:
            cube_size = self.cube_size

        center_xyz = self.center_xyz[index]
        center_uvd = self.joint3DToImg(center_xyz)

        gt3Dcrop = joint_xyz - center_xyz.reshape(1, 3)
        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size, self.img_size), self.paras)
        if self.phase == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
            imgD,_, curLabel,  cube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, self.cube_size, trans, mode, off, rot, sc, self.paras)
            curLabel = curLabel / (cube[2] / 2.0)
        else:
            imgD = self.normalize_img(depth_crop.max(),depth_crop, center_xyz, cube_size)
            curLabel = gt3Dcrop / (cube_size[2] / 2.0)
            cube = np.array(cube_size)
            com2D = center_uvd
            M = trans
        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        joint_uvd = torch.from_numpy(joint_img).float()
        joint_xyz = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()

        return data, joint_xyz, joint_uvd, center, M, cube

    def __len__(self):
        return len(self.all_joints_xyz)


class icvl_loader(loader):
    def __init__(self, root_dir, phase, aug_para=[10, 0.1, 180], cube_size=[200,200,200], img_size=128, joint_num=16,
                 center_type='refine', loader=icvl_reader, full_img=False):
        super(icvl_loader, self).__init__(root_dir, phase, img_size, center_type, 'icvl')
        self.paras = (240.99, 240.96, 160.0, 120.0)
        self.cube_size = cube_size
        self.flip = 1
        self.joint_num = joint_num
        self.full_img = full_img
        self.ori_img_size = (320, 240)

        self.img_size = img_size
        self.data_path = self.root_dir
        self.loader = loader

        self.all_joints_uvd, self.all_centers_xyz, self.img_dirs = self.read_joints(self.root_dir, self.phase)
        self.length = len(self.all_joints_uvd)
        self.aug_modes = ['rot', 'com', 'sc', 'none']#,'com','sc','none'
        self.aug_para = aug_para
        self.centers_xyz = self.all_centers_xyz

    def __getitem__(self, index):
        img_path = self.img_dirs[index]
        if not os.path.exists(img_path):
            index = index + 1
            img_path = self.img_dirs[index]
        depth = self.loader(img_path)

        joint_uvd = self.all_joints_uvd[index].copy()
        joint_xyz = self.jointImgTo3D(joint_uvd)

        center_xyz = self.centers_xyz[index].copy()
        center_uvd = self.joint3DToImg(center_xyz)
        gt3Dcrop = joint_xyz - center_xyz

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, self.cube_size,(self.img_size,self.img_size),self.paras)

        if self.phase == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
            imgD, _, curLabel, curCube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, self.cube_size, trans, mode, off, rot, sc,self.paras)
            curLabel = curLabel / (curCube[2] / 2.0)
        else:
            imgD = self.normalize_img(depth_crop.max(),depth_crop, center_xyz, self.cube_size)
            curLabel = gt3Dcrop / (self.cube_size[2] / 2.0)
            curCube = np.array(self.cube_size)
            com2D = center_uvd
            M = trans

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (curCube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size/2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (curCube[0] / 2.0)
        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        joint_uvd = torch.from_numpy(joint_img).float()
        joint_xyz = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(curCube).float()

        return data, joint_xyz, joint_uvd, center, M, cube

    def read_joints(self, data_rt,phase):
        if phase =='train':
            f = open(data_rt + "/train.txt", "r")
            f_center = open(data_rt+"/center_train_refined.txt")
            lines = f.readlines()
            lines_center = f_center.readlines()
            f.close()
        else:
            f1 = open(data_rt+"/test_seq_1.txt", "r")
            f2 = open(data_rt + "/test_seq_2.txt", "r")
            f_center = open(data_rt + "/center_test_refined.txt")
            lines = f1.read().splitlines()+f2.read().splitlines()
            while '' in lines:
                lines.remove('')
            lines_center = f_center.readlines()
            f1.close()
            f2.close()

        centers_xyz = []
        joints_uvd = []

        img_names = []
        subSeq = ['0']
        for index, line in enumerate(lines):
            strs = line.split()
            p = strs[0].split('/')
            if not self.full_img:
                if ('0' in subSeq) and len(p[0]) > 6:
                    pass
                elif not ('0' in subSeq) and len(p[0]) > 6:
                    continue
                elif (p[0] in subSeq) and len(p[0]) <= 6:
                    pass
                elif not (p[0] in subSeq) and len(p[0]) <= 6:
                    continue

            img_path = data_rt + '/Depth/' + strs[0]
            if not os.path.isfile(img_path):
                continue

            joint_uvd = np.array(list(map(float, strs[1:]))).reshape(16, 3)
            strs_center = lines_center[index].split()

            if strs_center[0] == 'invalid':
                continue
            else:
                center_xyz = np.array(list(map(float, strs_center))).reshape(3)

            centers_xyz.append(center_xyz)
            joints_uvd.append(joint_uvd)
            img_names.append(img_path)

        f_center.close()
        return joints_uvd, centers_xyz, img_names

    def __len__(self):
        return self.length


class msra_loader(loader):
    def __init__(self, root_dir, phase, aug_para=[10, 0.1, 180], img_size=128, joint_num=21, center_type='refine',
                 test_persons=[0], loader=msra_reader):
        super(msra_loader, self).__init__(root_dir, phase, img_size, center_type, 'msra')
        self.paras = (241.42, 241.42, 160, 120)
        self.cube_size = [200, 200, 200, 180, 180, 180, 170, 160, 150]
        self.centers_type = center_type
        self.aug_para = aug_para
        person_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        train_persons = list(set(person_list).difference(set(test_persons)))
        self.flip = -1
        if phase == 'train':
            self.all_joints_xyz, self.all_joints_uvd, self.keys = self.read_joints(root_dir, phase, persons=train_persons)
            self.length = len(self.all_joints_xyz)
        else:
            self.all_joints_xyz, self.all_joints_uvd, self.keys = self.read_joints(root_dir, phase, persons=test_persons)
            self.length = len(self.all_joints_xyz)
        file_uvd = open('./msra_label.txt', 'w')
        for index in range(len(self.all_joints_uvd)):
            np.savetxt(file_uvd, self.all_joints_uvd[index].reshape([1, joint_num * 3]), fmt='%.3f')
        if center_type == 'refine':
            file_name = self.root_dir + '/center_' + phase + '_' + str(test_persons[0]) + '_refined.txt'
            self.centers_xyz = np.loadtxt(file_name)

        self.loader = loader
        self.joint_num = joint_num
        self.aug_modes = ['rot', 'com', 'sc', 'none']

    def __getitem__(self, index):
        person = self.keys[index][0]
        name = self.keys[index][1]
        cube_size = [self.cube_size[person], self.cube_size[person], self.cube_size[person]]
        file = '%06d' % int(self.keys[index][2])

        depth, pcl = msra_reader(self.root_dir + "/P" + str(person) + "/" + str(name) + "/" + str(file) + "_depth.bin",
                                 self.paras)
        assert (depth.shape == (240, 320))

        joint_xyz = self.all_joints_xyz[index].copy()

        if self.center_type == 'refine':
            center_xyz = self.centers_xyz[index].copy()
            center_uvd = self.joint3DToImg(center_xyz)
        elif self.center_type == 'joint_mean':
            center_xyz = joint_xyz.mean(0)
            center_uvd = self.joint3DToImg(center_xyz)

        gt3Dcrop = joint_xyz - center_xyz

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size, self.img_size),self.paras)

        if self.phase == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1],
                                                   rot_range=self.aug_para[2])
            imgD, _, curLabel, curCube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, cube_size,
                                                                       trans, mode, off, rot, sc,self.paras)
            curLabel = curLabel / (curCube[2] / 2.0)
        else:
            imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, cube_size)
            curLabel = gt3Dcrop / (cube_size[2] / 2.0)
            curCube = np.array(cube_size)
            com2D = center_uvd
            M = trans
            sc = 1

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (curCube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (curCube[0] / 2.0)
        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        joint_img = torch.from_numpy(joint_img).float()
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(curCube).float()


        return data, joint, joint_img, center, M, cube


    # return joint_uvd
    def read_joints(self, data_rt, phase, persons=[0, 1, 2, 3, 4, 5, 6, 7],
                    poses=["1", "2", "3", "4", '5', '6', '7', '8', '9', 'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP', 'Y']):
        joints_xyz = []
        joints_uvd = []
        index = 0
        keys = {}
        file_record = open('./msra_record_list.txt', "w")
        for person in persons:
            for pose in poses:
                with open(data_rt + "/P" + str(person) + "/" + str(pose) + "/joint.txt") as f:
                    num_joints = int(f.readline())
                    for i in range(num_joints):
                        file_record.write(
                            'P' + str(person) + "/" + str(pose) + '/' + '%06d' % int(i) + "_depth.bin" + '\r\n')
                        joint = np.fromstring(f.readline(), sep=' ')
                        joint_xyz = joint.reshape(21, 3)
                        # need to chaneg z to -
                        joint_xyz[:, 2] = -joint_xyz[:, 2]
                        joint_uvd = self.joint3DToImg(joint_xyz)
                        # joint = joint.reshape(63)
                        joints_xyz.append(joint_xyz)
                        joints_uvd.append(joint_uvd)
                        keys[index] = [person, pose, i]
                        index += 1
        file_record.close()
        return joints_xyz, joints_uvd, keys

    def __len__(self):
        return self.length


class msra14_loader(loader):
    def __init__(self, root_dir, img_type, img_size=128, joint_num=21, center_type='mass', persons=[0,1,2,3,4,5], loader=msra14_reader):
        super(msra14_loader, self).__init__(root_dir, img_type, img_size, center_type, 'msra')
        self.paras = (241.42, 241.42, 160, 120)
        self.cube_size = [200, 180, 150, 180, 200, 180]
        self.centers_type = center_type
        self.flip = -1
        self.persons = persons
        self.all_joints_xyz, self.all_joints_uvd, self.keys = self.read_joints(root_dir, self.persons)
        self.length = len(self.all_joints_xyz)

        self.loader = loader
        self.joint_num = joint_num
        self.aug_modes = ['rot', 'com', 'sc', 'none']

    def __getitem__(self, index):
        person = self.keys[index][0]
        cube_size = [self.cube_size[person], self.cube_size[person], self.cube_size[person]]
        file = '%06d' % int(self.keys[index][1])

        depth = self.loader(self.root_dir + "/Subject" + str(person+1) + "/" + str(file) + "_depth.bin", self.paras)
        assert (depth.shape == (240, 320))

        joint_uvd = self.all_joints_uvd[index].copy()
        joint_xyz = np.zeros_like(joint_uvd)
        joint_xyz[:, 0], joint_xyz[:, 1] = pixel2world(joint_uvd[:, 0], joint_uvd[:, 1], joint_uvd[:, 2], self.paras)
        joint_xyz[:, 2] = joint_uvd[:, 2]

        if self.centers_type == 'mass':
            center_uvd = get_center_fast(depth)
            center_xyz = self.jointImgTo3D(center_uvd)
        elif self.centers_type == 'joint_mean':
            # center_uvd = joint_uvd.mean(0)
            # center_xyz = joint_xyz.mean(0)
            center_uvd = (joint_uvd[0,:]+joint_uvd[4,:]+joint_uvd[8,:]+joint_uvd[12,:]+joint_uvd[16,:]+joint_uvd[20,:])/6.0
            center_xyz = (joint_xyz[0,:]+joint_xyz[4,:]+joint_xyz[8,:]+joint_xyz[12,:]+joint_xyz[16,:]+joint_xyz[20,:])/6.0

        gt3Dcrop = joint_xyz - center_xyz

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size, self.img_size),self.paras)

        imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, cube_size)
        curLabel = gt3Dcrop / (cube_size[2] / 2.0)
        curCube = np.array(cube_size)
        com2D = center_uvd
        M = trans

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (curCube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (curCube[0] / 2.0)

        pcl = self.getpcl(imgD, center_xyz, curCube, M)
        pcl_index = np.arange(pcl.shape[0])
        pcl_num = pcl.shape[0]
        if pcl_num == 0:
            pcl_sample = np.zeros([self.sample_num, 3])
        else:
            if pcl_num < self.sample_num:
                tmp = math.floor(self.sample_num / pcl_num)
                index_temp = pcl_index.repeat(tmp)
                pcl_index = np.append(index_temp,
                                      np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1],
                                                       replace=False))
            select = np.random.choice(pcl_index, self.sample_num, replace=False)
            pcl_sample = pcl[select, :]

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)


        pcl_sample = torch.from_numpy(pcl_sample.transpose(1, 0)).float()
        joint_img = torch.from_numpy(joint_img).float()
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(curCube).float()

        so_node = torch.ones([self.joint_num,3])
        visible = torch.zeros([self.joint_num])
        outline = torch.zeros([self.joint_num])

        return data, pcl_sample, joint, joint_img, so_node, center, M, cube,visible,outline

    # return joint_uvd
    def read_joints(self, data_rt, persons):
        joints_xyz = []
        joints_uvd = []
        index = 0
        keys = {}
        for person in persons:
            with open(data_rt + "/Subject" + str(person+1) + "/joint.txt") as f:
                num_joints = int(f.readline())
                for i in range(num_joints):
                    joint = np.fromstring(f.readline(), sep=' ')
                    joint_xyz = joint.reshape(21, 3)
                    joint_xyz[:, 2] = -joint_xyz[:, 2]
                    joint_uvd = np.zeros_like(joint_xyz)
                    # need to chaneg z to -
                    joint_uvd[:, 0], joint_uvd[:, 1] = world2pixel(joint_xyz[:, 0], joint_xyz[:, 1],
                                                                   joint_xyz[:, 2], self.paras)
                    joint_uvd[:, 2] = joint_xyz[:, 2]
                    joints_xyz.append(joint_xyz)
                    joints_uvd.append(joint_uvd)
                    keys[index] = [person, i]
                    index += 1
        return joints_xyz, joints_uvd, keys

    def __len__(self):
        return self.length


class hands17_loader(loader):
    def __init__(self, root_dir, phase, aug_para=[0, 0, 0], img_size=128, center_type='refine',
                 cube_size=[200, 200, 200], joint_num=21, img_num=957032, loader=hands17_reader):
        super(hands17_loader, self).__init__(root_dir, phase, img_size, '', 'HANDS17')
        self.paras = (475.065948, 475.065857, 315.944855, 245.287079)
        self.aug_para = aug_para
        self.cube_size = cube_size
        self.flip = 1
        self.joint_num = joint_num
        self.img_num = img_num

        self.img_size = img_size
        self.data_path = self.root_dir
        self.loader = loader

        print('loading data...')
        self.all_joints_xyz, self.all_joints_uvd, self.all_centers_xyz, self.all_centers_uvd, self.img_dirs = self.read_joints(self.root_dir)
        print('finish!!')
        self.length = len(self.all_joints_xyz)
        self.aug_modes = ['com','sc','none','rot']#,'com','sc','none''rot','com',
        self.center_type = center_type
        print('aug_mode', self.aug_modes)

    def __getitem__(self, index):
        img_path = self.img_dirs[index]
        if not os.path.exists(img_path):
            index = index + 1
            img_path = self.img_dirs[index]
        depth = self.loader(img_path)

        joint_xyz = self.all_joints_xyz[index].copy()
        center_uvd = self.all_centers_uvd[index].copy()
        center_xyz = self.all_centers_xyz[index].copy()

        gt3Dcrop = joint_xyz - center_xyz

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, self.cube_size,(self.img_size,self.img_size),self.paras)

        if self.phase == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])#10, 0.1, 180
            imgD, _, curLabel, cube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, self.cube_size, trans, mode, off, rot, sc, self.paras)
            curLabel = curLabel / (cube[2] / 2.0)
        else:
            imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, self.cube_size)
            curLabel = gt3Dcrop / (self.cube_size[2] / 2.0)
            cube = np.array(self.cube_size)
            com2D = center_uvd
            M = trans

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size/2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        joint_uvd = torch.from_numpy(joint_img).float()
        joint_xyz = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()

        return data, joint_xyz, joint_uvd, center, M, cube

    def read_joints(self, data_rt):
        centers_xyz = []
        centers_uvd = []
        joints_xyz = []
        joints_uvd = []
        img_names = []
        if self.phase =='train':
            f = open(data_rt + "/training/Training_Annotation.txt", "r")
            f_center = open(data_rt+"/center_train_refined.txt")
            lines = f.readlines()
            lines_center = f_center.readlines()
            f.close()
            f_center.close()

            for index, line in enumerate(lines):
                if index > self.img_num:
                    break
                strs = line.split()
                img_path = data_rt + '/training/images/' + strs[0]
                if not os.path.isfile(img_path):
                    continue

                joint_xyz = np.array(list(map(float, strs[1:]))).reshape(self.joint_num, 3)
                strs_center = lines_center[index].split()

                if strs_center[0] == 'invalid':
                    continue
                else:
                    center_xyz = np.array(list(map(float, strs_center))).reshape(3)
                center_uvd = self.joint3DToImg(center_xyz)
                centers_xyz.append(center_xyz)
                centers_uvd.append(center_uvd)

                joint_uvd = self.joint3DToImg(joint_xyz)
                joints_xyz.append(joint_xyz)
                joints_uvd.append(joint_uvd)
                img_names.append(img_path)
        else:
            f = open(data_rt+"/frame/BoundingBox.txt", "r")
            f_center = open(data_rt + "/center_test_refined.txt")
            lines = f.readlines()
            lines_center = f_center.readlines()
            f.close()
            f_center.close()
            for index, line in enumerate(lines):
                if index > self.img_num:
                    break
                strs = line.split()
                img_path = data_rt + '/frame/images/' + strs[0]
                if not os.path.isfile(img_path):
                    continue
                strs_center = lines_center[index].split()
                if strs_center[0] == 'invalid':
                    continue
                else:
                    center_xyz = np.array(list(map(float, strs_center))).reshape(3)
                center_uvd = self.joint3DToImg(center_xyz)
                centers_xyz.append(center_xyz)
                centers_uvd.append(center_uvd)

                joints_xyz.append(np.ones([self.joint_num, 3]))
                joints_uvd.append(np.ones([self.joint_num, 3]))
                img_names.append(img_path)

        return joints_xyz, joints_uvd, centers_xyz, centers_uvd, img_names

    def __len__(self):
        return self.length


if __name__ == "__main__":
    GFM = generateFeature.GFM()

    # root = 'D:\\dataset\\nyu'
    # root = 'D:\\dataset\\msra14\\'
    # root = 'D:\\dataset\\msra\\'
    root = 'D:\\dataset\\icvl\\'

    # synthetic_dataset = nyu_loader(root, 'train', center_type='refine', aug_para=[10, 0.1, 180])
    # synthetic_dataset = msra_loader(root, 'train', center_type='refine', aug_para=[10, 0.1, 180])
    synthetic_dataset = icvl_loader(root, 'train', center_type='joint_mean', aug_para=[0, 0, 0])
    print(synthetic_dataset.__len__())

    dataloader = DataLoader(synthetic_dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=False)

    for index, data in enumerate(dataloader):
        # ind_num, img = data
        img, j3d_xyz, j3d_uvd, center, M, cube = data
        # vis_tool.debug_ThreeView_pose(pcl, j3d_xyz, index, 'msra', './debug','view')
        # vis_tool.debug_2d_heatmap(img, heatmap, index, './debug','m2p')
        vis_tool.debug_2d_pose(img, j3d_uvd, index, 'icvl', './debug', 'img', 32)

        print(index)
        if index == 3:
            break

    print('done')



