# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from vis_tool import *
from sklearn.decomposition import PCA

xrange = range


def get_center_adopt(img):
    img_dim = img.reshape(-1)
    min_value = np.sort(img_dim)[:300].mean()
    centers = np.array([0.0, 0.0, 300.0])
    flag = np.logical_and(img <= min_value + 100, img >= min_value)
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


def getCrop(depth, xstart, xend, ystart, yend, zstart, zend, thresh_z=True, background=0):
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
        cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0),
                                       abs(yend)-min(yend, depth.shape[0])),
                                      (abs(xstart)-max(xstart, 0),
                                       abs(xend)-min(xend, depth.shape[1]))), mode='constant', constant_values=background)
    elif len(depth.shape) == 3:
        cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1]), :].copy()
        # add pixels that are out of the image in order to keep aspect ratio
        cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0),
                                       abs(yend)-min(yend, depth.shape[0])),
                                      (abs(xstart)-max(xstart, 0),
                                       abs(xend)-min(xend, depth.shape[1])),
                                      (0, 0)), mode='constant', constant_values=background)
    else:
        raise NotImplementedError()

    if thresh_z is True:
        msk1 = np.logical_and(cropped < zstart, cropped != 0)
        msk2 = np.logical_and(cropped > zend, cropped != 0)
        cropped[msk1] = zstart
        cropped[msk2] = 0.  # backface is at 0, it is set later
    return cropped


def nyu_reader(img_path):
    img = cv2.imread(img_path)
    depth = np.asarray(img[:, :, 0] + img[:, :, 1] * 256, dtype=np.float32)
    return depth


def icvl_reader(img_path):
    img = Image.open(img_path)  # open image
    assert len(img.getbands()) == 1  # ensure depth image
    depth = np.asarray(img, np.float32)
    return depth


def msra_reader(image_name,para):
    f = open(image_name, 'rb')
    data = np.fromfile(f, dtype=np.uint32)
    width, height, left, top, right , bottom = data[:6]
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


def hands17_reader(img_path):
    img = Image.open(img_path)  # open image
    assert len(img.getbands()) == 1  # ensure depth image
    depth = np.asarray(img, np.float32)
    return depth


def get_center(img, upper=1000, lower=10):
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


class loader(Dataset):
    def __init__(self, root_dir, img_type, img_size, dataset_name):
        self.rng = np.random.RandomState(23455)
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.img_type = img_type
        self.img_size = img_size
        self.allJoints = False
        # create OBB
        self.pca = PCA(n_components=3)
        self.sample_num = 1024

    # joint position from image to world
    def jointImgTo3D(self, uvd):
        fx, fy, fu, fv = self. paras
        ret = np.zeros_like(uvd, np.float32)
        if len(ret.shape) == 1:
            ret[0] = (uvd[0] - fu) * uvd[2] / fx
            ret[1] = self.flip * (uvd[1] - fv) * uvd[2] / fy
            ret[2] = uvd[2]
        elif len(ret.shape) == 2:
            ret[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
            ret[:, 1] = self.flip * (uvd[:, 1] - fv) * uvd[:, 2] / fy
            ret[:, 2] = uvd[:, 2]
        else:
            ret[:, :, 0] = (uvd[:, :, 0] - fu) * uvd[:, :, 2] / fx
            ret[:, :, 1] = self.flip * (uvd[:, :, 1] - fv) * uvd[:, :, 2] / fy
            ret[:, :, 2] = uvd[:, :, 2]

        return ret

    def joint3DToImg(self,xyz):
        fx, fy, fu, fv = self.paras
        ret = np.zeros_like(xyz, np.float32)
        if len(ret.shape) == 1:
            ret[0] = (xyz[0] * fx / xyz[2] + fu)
            ret[1] = (self.flip * xyz[1] * fy / xyz[2] + fv)
            ret[2] = xyz[2]
        elif len(ret.shape) == 2:
            ret[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
            ret[:, 1] = (self.flip * xyz[:, 1] * fy / xyz[:, 2] + fv)
            ret[:, 2] = xyz[:, 2]
        else:
            ret[:, :, 0] = (xyz[:, :, 0] * fx / xyz[:, :, 2] + fu)
            ret[:, :, 1] = (self.flip * xyz[:, :, 1] * fy / xyz[:, :, 2] + fv)
            ret[:, :, 2] = xyz[:, :, 2]
        return ret

    def comToBounds(self, com, size):
        fx, fy, fu, fv = self.paras
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        xstart = int(np.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2] * fx + 0.5))
        xend = int(np.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2] * fx + 0.5))
        ystart = int(np.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2] * fy + 0.5))
        yend = int(np.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2] * fy + 0.5))
        return xstart, xend, ystart, yend, zstart, zend

    def comToTransform(self, com, size, dsize):
        """
        Calculate affine transform from crop
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :return: affine transform
        """

        xstart, xend, ystart, yend, _, _ = self.comToBounds(com, size)

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

    def recropHand(self, crop, M, Mnew, target_size, background_value=0., nv_val=0., thresh_z=True, com=None,
                   size=(250, 250, 250)):

        flags = cv2.INTER_LINEAR

        warped = cv2.warpPerspective(crop, np.dot(M, Mnew), target_size, flags=flags,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=float(background_value))
        warped[np.isclose(warped, nv_val)] = background_value

        if thresh_z is True:
            assert com is not None
            _, _, _, _, zstart, zend = self.comToBounds(com, size)
            msk1 = np.logical_and(warped < zstart, warped != 0)
            msk2 = np.logical_and(warped > zend, warped != 0)
            warped[msk1] = zstart
            warped[msk2] = 0.  # backface is at 0, it is set later

        return warped
    def moveCoM(self, dpt, cube, com, off, joints3D, M, pad_value=0):
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
            Mnew = self.comToTransform(new_com, cube, dpt.shape)
            new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, background_value=pad_value,
                                 nv_val=32000., thresh_z=True, com=new_com, size=cube)
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

        flags = cv2.INTER_LINEAR

        new_dpt = cv2.warpAffine(dpt, M, (dpt.shape[1], dpt.shape[0]), flags=flags,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=pad_value)

        com3D = self.jointImgTo3D(com)
        joint_2D = self.joint3DToImg(joints3D + com3D)
        data_2D = np.zeros_like(joint_2D)
        for k in xrange(data_2D.shape[0]):
            data_2D[k] = rotatePoint2D(joint_2D[k], com[0:2], rot)
        new_joints3D = (self.jointImgTo3D(data_2D) - com3D)

        return new_dpt, new_joints3D, rot

    def scaleHand(self, dpt, cube, com, sc, joints3D, M, pad_value=0):
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
            Mnew = self.comToTransform(com, new_cube, dpt.shape)
            new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, background_value=pad_value,
                                 nv_val=32000., thresh_z=True, com=com, size=cube)
        else:
            Mnew = M
            new_dpt = dpt

        new_joints3D = joints3D

        return new_dpt, new_joints3D, new_cube, Mnew

    def rand_augment(self, sigma_com=None, sigma_sc=None, rot_range=None):
        if sigma_com is None:
            sigma_com = 35.

        if sigma_sc is None:
            sigma_sc = 0.05

        if rot_range is None:
            rot_range = 180.

        mode = self.rng.randint(0, len(self.aug_modes))
        off = self.rng.randn(3) * sigma_com  # +-px/mm
        rot = self.rng.uniform(-rot_range, rot_range)
        sc = abs(1. + self.rng.randn() * sigma_sc)

        return mode, off, rot, sc

    def augmentCrop(self,img, gt3Dcrop, com, cube, M, mode, off, rot, sc, normZeroOne=False):
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
        if self.aug_modes[mode] == 'com':
            rot = 0.
            sc = 1.
            imgD, new_joints3D, com, M = self.moveCoM(img.astype('float32'), cube, com, off, gt3Dcrop, M, pad_value=0)
        elif self.aug_modes[mode] == 'rot':
            off = np.zeros((3,))
            sc = 1.
            imgD, new_joints3D, rot = self.rotateHand(img.astype('float32'), cube, com, rot, gt3Dcrop, pad_value=0)
        elif self.aug_modes[mode] == 'sc':
            off = np.zeros((3,))
            rot = 0.
            imgD, new_joints3D, cube, M = self.scaleHand(img.astype('float32'), cube, com, sc, gt3Dcrop, M, pad_value=0)
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

    def normalize_img(self, premax, imgD, com, cube):
        imgD[imgD == premax] = com[2] + (cube[2] / 2.)
        imgD[imgD == 0] = com[2] + (cube[2] / 2.)
        imgD[imgD >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
        imgD[imgD <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
        imgD -= com[2]
        imgD /= (cube[2] / 2.)
        return imgD

    # use deep-pp's method
    def Crop_Image_deep_pp(self, depth, com, size, dsize):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """

        if len(size) != 3 or len(dsize) != 2:
            raise ValueError("Size must be 3D and dsize 2D bounding box")

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size)

        # crop patch from source
        cropped = getCrop(depth, xstart, xend, ystart, yend, zstart, zend)

        # resize to same size
        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], hb * dsize[0] // wb)
        else:
            sz = (wb * dsize[1] // hb, dsize[1])

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


    def unnormal_joint_img(self, joint_img):
        device = joint_img.device
        joint = torch.zeros(joint_img.size()).to(device)
        joint[:, :, 0:2] = (joint_img[:, :, 0:2] + 1)/2 * self.img_size
        joint[:, :, 2] = (joint_img[:, :, 2] + 1)/2 * self.cube_size[2]
        return joint

    def jointsImgTo3D(self, joint_uvd):
        joint_xyz = torch.zeros_like(joint_uvd)
        joint_xyz[:, :, 0] = (joint_uvd[:, :, 0]-self.paras[2])*joint_uvd[:, :, 2]/self.paras[0]
        joint_xyz[:, :, 1] = self.flip * (joint_uvd[:, :, 1]-self.paras[3])*joint_uvd[:, :, 2]/self.paras[1]
        joint_xyz[:, :, 2] = joint_uvd[:, :, 2]
        return joint_xyz

    def joints3DToImg(self, joint_xyz):
        fx, fy, fu, fv = self.paras
        joint_uvd = torch.zeros_like(joint_xyz)
        joint_uvd[:, :, 0] = (joint_xyz[:, :, 0] * fx / joint_xyz[:,:, 2] + fu)
        joint_uvd[:, :, 1] = (self.flip * joint_xyz[:, :, 1] * fy / joint_xyz[:, :, 2] + fv)
        joint_uvd[:, :, 2] = joint_xyz[:, :, 2]
        return joint_uvd

    def uvd_nl2xyznl_tensor(self, joint_uvd, M, cube_size, center):
        batch_size, joint_num, _ = joint_uvd.size()
        device = joint_uvd.device
        joint_img = torch.zeros_like(joint_uvd)
        cube_size_t = cube_size.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        M_t = M.to(device).view(batch_size, 1, 3, 3).repeat(1, joint_num, 1, 1)
        M_inverse = torch.inverse(M_t)
        joint_img[:, :, 0:2] = (joint_uvd[:, :, 0:2] + 1) * (self.img_size/2)
        joint_img[:, :, 2] = (joint_uvd[:, :, 2]) * (cube_size_t[:, :, 2] / 2.0) + center_t[:, :, 2]
        joint_uvd = self.get_trans_points(joint_img, M_inverse)
        joint_xyz = self.jointsImgTo3D(joint_uvd)
        joint_xyz = (joint_xyz - center_t) / (cube_size_t / 2.0)
        return joint_xyz

    def xyz_nl2uvdnl_tensor(self, joint_xyz, M, cube_size, center):
        device = joint_xyz.device
        batch_size, joint_num, _ = joint_xyz.size()
        joint_world = torch.zeros_like(joint_xyz)
        cube_size_t = cube_size.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        M_t = M.to(device).view(batch_size, 1, 3, 3).repeat(1, joint_num, 1, 1)

        joint_temp = joint_xyz * cube_size_t / 2.0 + center_t
        joint_uvd = self.joints3DToImg(joint_temp)
        joint_uvd= self.get_trans_points(joint_uvd, M_t)
        joint_world[:,:,0:2] = joint_uvd[:,:,0:2]/self.img_size*2 - 1
        joint_world[:,:,2] = (joint_uvd[:,:,2] - center_t[:,:,2]) / (cube_size_t[:,:,2]/2)
        return joint_world

    def get_trans_points(self, joints, M):
        device = joints.device
        joints_trans = torch.zeros_like(joints)
        joints_mat = torch.cat((joints[:, :, 0:2], torch.ones(joints.size(0), joints.size(1), 1).to(device)), dim=-1)
        joints_trans[:, :, 0:2] = torch.matmul(M, joints_mat.unsqueeze(-1)).squeeze(-1)[:, :, 0:2]
        joints_trans[:, :, 2] = joints[:, :, 2]
        return joints_trans


class realtime_loader(loader):
    def __init__(self, data_path, paras, frame_len, img_size=128, cube_size=[300, 300, 300]):
        super(realtime_loader, self).__init__('', 'test', img_size, 'realtime')
        self.data_path = data_path
        self.cube_size = cube_size
        self.frame_len = frame_len
        self.paras = paras
        self.flip = 1

    def __getitem__(self, index):
        img_path = self.data_path + '/'+str(index+1).zfill(8)+'.png'
        # img_path = self.data_path + '/'+str(index)+'.png'
        image = cv2.imread(img_path, -1)
        depth = np.asarray(image, dtype=np.float32)

        depth = depth[:, ::-1]
        depth[depth == 0] = depth.max()

        # center_uvd = get_center_fast(depth)
        center_uvd = get_center_adopt(depth)
        center_xyz = self.jointImgTo3D(center_uvd)
        cube_size = self.cube_size

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, dsize=(self.img_size, self.img_size))

        imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, cube_size)
        curCube = np.array(cube_size)
        com2D = center_uvd
        M = trans

        com3D = self.jointImgTo3D(com2D)

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)
        center = torch.from_numpy(com3D).float()
        center_uvd = torch.from_numpy(center_uvd).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(curCube).float()
        return data, center, center_uvd, M, cube

    def __len__(self):
        return self.frame_len


def RunMyImageFloder():
    cuda_id = 0
    torch.cuda.set_device(cuda_id)

    test_data = realtime_loader('D:\BaiduNetdiskDownload\SRN\data\kinect2', (363.9, 363.9, 255.4, 206.3), 300, cube_size=[250,250,250])
    dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)

    for index, data in enumerate(dataloader):
        img, center, center_uvd, M, cube = data
        img = (img.view(128,128).numpy() + 1)/2 *255
        cv2.imwrite('./debug/'+str(index)+'.png', img)
        print(index)


if __name__ == "__main__":
    RunMyImageFloder()