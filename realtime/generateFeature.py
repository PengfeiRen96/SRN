import numpy as np
import torch
from numpy import linalg
import torch.nn.functional as F


def Matr(axis, theta):
    device = theta.device
    batchsize = theta.size()[0]
    M = torch.eye(4, requires_grad=False).repeat(batchsize, 1, 1)
    if axis == 0:
        M[:, 1, 1] = torch.cos(theta)
        M[:, 1, 2] = -torch.sin(theta)
        M[:, 2, 1] = torch.sin(theta)
        M[:, 2, 2] = torch.cos(theta)
    elif axis == 1:
        M[:, 0, 0] = torch.cos(theta)
        M[:, 0, 2] = -torch.sin(theta)
        M[:, 2, 0] = torch.sin(theta)
        M[:, 2, 2] = torch.cos(theta)
    elif axis == 2:
        M[:, 0, 0] = torch.cos(theta)
        M[:, 0, 1] = -torch.sin(theta)
        M[:, 1, 0] = torch.sin(theta)
        M[:, 1, 1] = torch.cos(theta)
    else:
        M[:, axis - 3, 3] = theta
    return M.to(device)

#generate feature module
class GFM:
    def __init__(self):
        self.edt_size = 32
        self.offset_theta = 1

    def rotation_points(self, points, rot):
        device = points.device
        points_rot = points.view(points.size(0), -1, 3)
        points_rot = torch.cat((points_rot, torch.ones(points_rot.size(0), points_rot.size(1), 1).to(device)),dim=-1)
        theta_x = torch.tensor(rot[:, 0]).float().to(device)
        theta_y = torch.tensor(rot[:, 1]).float().to(device)
        theta_z = torch.tensor(rot[:, 2]).float().to(device)

        points_rot = torch.matmul(points_rot, Matr(2, theta_z))
        points_rot = torch.matmul(points_rot, Matr(0, theta_x))
        points_rot = torch.matmul(points_rot, Matr(1, theta_y))

        return points_rot[:, :, 0:3]

    def pcl2img(self, pcl, img_size):
        divce = pcl.device
        pcl = pcl.permute(0,2,1)
        img_pcl = torch.ones([pcl.size(0), img_size*img_size]).to(divce) * -1
        index_x = torch.clamp(torch.floor((pcl[:, :, 0] + 1) / 2 * img_size), 0, img_size - 1).long().to(divce)
        index_y = torch.clamp(torch.floor((1 - pcl[:, :, 1]) / 2 * img_size), 0, img_size - 1).long().to(divce)*img_size
        index = index_y + index_x
        batch_index = torch.arange(pcl.size(0)).unsqueeze(-1).expand(-1, index.size(1))
        img_pcl[batch_index, index] = (pcl[:, :, 2] + 1) / 2
        return img_pcl.view(pcl.size(0), 1, img_size, img_size)

    def joint2heatmap2d(self, joint, isFlip=True, std = 3.4, heatmap_size=32):
        # joint depth is norm[-1,1]
        if isFlip:
            coeff = -1
        else:
            coeff = 1
        divce = joint.device
        batch_size, joint_num, _ = joint.size()
        xx, yy = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size))
        mesh_x = torch.from_numpy(xx).view(1, 1, heatmap_size, heatmap_size).repeat(batch_size, joint_num, 1, 1).float().to(divce)
        mesh_y = torch.from_numpy(yy).view(1, 1, heatmap_size, heatmap_size).repeat(batch_size, joint_num, 1, 1).float().to(divce)
        joint_ht = torch.zeros_like(joint).to(divce)
        joint_ht[:, :, 0] = (joint[:, :, 0] + 1) / 2 * heatmap_size
        joint_ht[:, :, 1] = (coeff*joint[:, :, 1] + 1) / 2 * heatmap_size
        joint_x = joint_ht[:, :, 0].view(batch_size, joint_num, 1, 1).repeat(1, 1, heatmap_size, heatmap_size).float()
        joint_y = joint_ht[:, :, 1].view(batch_size, joint_num, 1, 1).repeat(1, 1, heatmap_size, heatmap_size).float()
        heatmap = torch.exp(-(torch.pow((mesh_x - joint_x) / std, 2) + torch.pow((mesh_y.to(divce) - joint_y) / std, 2)))
        return heatmap

    def joint2heatmap3d(self, joint, isFlip=True,heatmap_size=32):
        # joint depth is norm[-1,1]
        if isFlip:
            coeff = -1
        else:
            coeff = 1
        std = 1.2
        batch_size, joint_num, _ = joint.size()
        joint_ht = torch.zeros_like(joint)
        joint_ht[:, :, 0] = (joint[:, :, 0] + 1) / 2 * heatmap_size
        joint_ht[:, :, 1] = (coeff*joint[:, :, 1] + 1) / 2 * heatmap_size
        joint_ht[:, :, 2] = (joint[:, :, 2] + 1) / 2 * heatmap_size
        joint_x = joint_ht[:, :, 0].view(batch_size, joint_num, 1, 1, 1).repeat(1, 1, heatmap_size, heatmap_size, heatmap_size).float()
        joint_y = joint_ht[:, :, 1].view(batch_size, joint_num, 1, 1, 1).repeat(1, 1, heatmap_size, heatmap_size, heatmap_size).float()
        joint_z = joint_ht[:, :, 2].view(batch_size, joint_num, 1, 1, 1).repeat(1, 1, heatmap_size, heatmap_size, heatmap_size).float()
        # joint_x = joint_ht[:, :, 0].view(batch_size, joint_num, 1, 1, 1).expand(-1, -1, heatmap_size, heatmap_size, heatmap_size).float()
        # joint_y = joint_ht[:, :, 1].view(batch_size, joint_num, 1, 1, 1).expand(-1, -1, heatmap_size, heatmap_size, heatmap_size).float()
        # joint_z = joint_ht[:, :, 2].view(batch_size, joint_num, 1, 1, 1).expand(-1, -1, heatmap_size, heatmap_size, heatmap_size).float()
        heatmap = torch.exp(-(torch.pow((self.mesh_x3d - joint_x) / std, 2) + torch.pow((self.mesh_y3d - joint_y) / std, 2) + torch.pow((self.mesh_z3d - joint_z) / std, 2)))
        return heatmap

    def joint2offset(self,joint,img,feature_size=32):
        device = joint.device
        batch_size,_,img_height,img_width = img.size()
        img = F.interpolate(img,size=[feature_size,feature_size])
        _,joint_num,_ = joint.view(batch_size,-1,3).size()
        joint_feature = joint.view(joint.size(0),-1,1,1).repeat(1,1,feature_size,feature_size)
        mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_y,mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords,img),dim=1).repeat(1, joint_num, 1, 1)
        offset = joint_feature - coords
        offset = offset.view(batch_size,joint_num,3,feature_size,feature_size)
        dist = torch.sqrt(torch.sum(torch.pow(offset,2),dim=2)+1e-8)
        offset_norm = (offset / (dist.unsqueeze(2)))
        heatmap = self.offset_theta-dist
        # heatmap = - dist
        mask = heatmap.ge(0).float() * img.lt(1).float().view(batch_size,1,feature_size,feature_size)
        offset_norm_mask = (offset_norm*mask.unsqueeze(2)).view(batch_size,-1,feature_size,feature_size)
        heatmap_mask = heatmap * mask.float()
        return torch.cat((offset_norm_mask,heatmap_mask),dim=1)

    def offset2joint(self, offset, depth):
        device = offset.device
        batch_size,joint_num,feature_size,feature_size = offset.size()
        joint_num = joint_num / 4
        if depth.size(-1)!=feature_size:
            depth = F.interpolate(depth, size=[feature_size, feature_size])
        offset_unit = offset[:,:joint_num*3,:,:].contiguous().view(batch_size,joint_num,3,-1)
        heatmap = offset[:,joint_num*3:,:,:].contiguous().view(batch_size,joint_num,-1)
        mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_y,mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords,depth),dim=1).repeat(1, joint_num, 1, 1).view(batch_size,joint_num,3,-1)
        value,index = torch.topk(heatmap,30,dim=-1)
        index = index.unsqueeze(2).repeat(1,1,3,1)
        value = value.unsqueeze(2).repeat(1,1,3,1)
        offset_unit_select = torch.gather(offset_unit,-1,index)
        coords_select = torch.gather(coords,-1,index)
        dist = self.offset_theta-value
        joint = torch.sum((offset_unit_select*dist + coords_select)*value,dim=-1)
        joint = joint / torch.sum(value,-1)
        return joint

    # output size : B*4(xyz+point_type)*N
    def joint2pc(self, joint, seed = 12345, sample_point=1024, radius=0.08):
        device = joint.device
        batch_size, joint_num, _ = joint.size()

        radius = torch.rand([batch_size, joint_num, 100]).to(device)*radius
        theta = torch.rand([batch_size, joint_num, 100]).to(device)*np.pi
        phi = torch.rand([batch_size, joint_num, 100]).to(device)*np.pi

        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * torch.cos(phi)
        type = torch.arange(1, joint_num+1).float().to(device).view(1, joint_num, 1).repeat(batch_size, 1, 100)

        point = joint.unsqueeze(-2).repeat(1,1,100,1) + torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)), dim = -1)
        point = torch.cat((point, type.unsqueeze(-1)), dim=-1)
        point = point.view(batch_size,-1,4)
        sample = np.random.choice(point.size(1), sample_point, replace=False)
        return point[:, sample, :].permute(0, 2, 1)

    # joint B*J*3    pointcloud B*3*N
    def joint2heatmap_pcl(self, joint, pointcloud, cube_size):
        joint = joint.view(-1, self.joint_num, 3)
        batch_size, joint_num, _ = joint.size()
        sample_num = self.sample_num
        radius = 80.0 / cube_size
        pcl = pointcloud.permute(0, 2, 1).contiguous()
        point_heatmap = torch.zeros([batch_size, joint_num, sample_num]).float().cuda()
        point_unit = torch.zeros([batch_size, joint_num, sample_num, 3]).float().cuda()
        heatmap_batch_index = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, joint_num,self.knn_num).long()
        heatmap_joint_index = torch.arange(0, joint_num).view(1, joint_num, 1).repeat(batch_size, 1,self.knn_num).long()
        unit_batch_index = torch.arange(0, batch_size).view(batch_size, 1, 1, 1).repeat(1, 1, self.knn_num, 3).long()
        unit_joint_index = torch.arange(0, joint_num).view(1, joint_num, 1, 1).repeat(batch_size, 1, self.knn_num,3).long()
        unit_xyz_index = torch.arange(0, 3).view(1, 1, 1, 3).repeat(batch_size, joint_num, self.knn_num, 1).long()

        # use torch's method
        inputs_diff = pcl.transpose(1, 2).unsqueeze(1).expand(batch_size, joint_num, 3, sample_num) \
                      - joint.unsqueeze(-1).expand(batch_size, joint_num, 3, sample_num)  # B * J * 3 * 1024
        inputs_diff = torch.mul(inputs_diff, inputs_diff)  # B * J * 3 * 1024
        inputs_diff = inputs_diff.sum(2)  # B * J * 1024
        dists, idx = torch.topk(inputs_diff, self.knn_num, 2, largest=False,sorted=True)  # dists: B * 512 * 64; inputs1_idx: B * 512 * 64
        # ball query
        invalid_map = dists.gt(radius.view(-1, 1, 1).expand(batch_size, joint_num, self.knn_num))  # B * J * 64
        for jj in range(self.joint_num):
            idx[:, jj, :][invalid_map[:, jj, :]] = jj

        idx_unit = idx.view(batch_size, -1, 1).repeat(1, 1, 3)
        inputs_level = pcl.gather(1, idx_unit).view(batch_size, joint_num, self.knn_num, 3)
        grouped_pcl = joint.unsqueeze(-2).repeat(1, 1, self.knn_num, 1) - inputs_level  # B * J * knn * 3
        grouped_len = grouped_pcl.pow(2).sum(-1).sqrt()

        point_heatmap[heatmap_batch_index, heatmap_joint_index, idx] = 1 - grouped_len / radius.view(-1, 1, 1).expand(batch_size,joint_num, self.knn_num)
        idx_unit = idx.unsqueeze(-1).repeat(1, 1, 1, 3)
        point_unit[unit_batch_index, unit_joint_index, idx_unit, unit_xyz_index] = grouped_pcl / ( grouped_len.unsqueeze(-1).repeat(1, 1, 1, 3))
        return torch.cat((point_heatmap, point_unit.permute(0, 1, 3, 2).contiguous().view(batch_size, -1, self.sample_num)), dim=1)

    def heatmap2joint_pcl(self, estimation, points, cube_size):
        radius = 80.0 / cube_size
        batch_size = estimation.size()[0]
        sample_num = self.sample_num
        joint_num = self.joint_num

        unit_batch_index = torch.arange(0, batch_size).view(batch_size, 1, 1, 1).repeat(1, 1, self.knn_num, 3).long()
        unit_joint_index = torch.arange(0, joint_num).view(1, joint_num, 1, 1).repeat(batch_size, 1, self.knn_num,
                                                                                      3).long()
        unit_xyz_index = torch.arange(0, 3).view(1, 1, 1, 3).repeat(batch_size, joint_num, self.knn_num, 1).long()

        outputs_xyz = estimation.data  # B*(4*self.joint_num)*sample_num
        output_heatmaps = outputs_xyz[:, 0:joint_num, :]  # B*self.joint_num*sample_num
        output_units = outputs_xyz[:, joint_num:, :].contiguous().view(batch_size,joint_num, 3, sample_num).permute(0, 1, 3, 2)  # B*self.joint_num*sample_num*3
        heatmap_select, point_index = torch.topk(output_heatmaps, self.knn_num, 2)  # B*self.joint_num*knn_num
        heatmap_select = heatmap_select + 1e-8
        unit_select = output_units[unit_batch_index, unit_joint_index, point_index.unsqueeze(-1).repeat(1, 1, 1, 3), unit_xyz_index]  # B*self.joint_num*knn_num*3
        vec_heatmap = ((1 - heatmap_select) * radius.view(-1, 1, 1).expand(batch_size, joint_num,self.knn_num)).unsqueeze(-1).repeat(1, 1, 1,3)  # B*self.joint_num*knn_num*3
        vec_predict = unit_select * vec_heatmap  # B*(self.joint_num)*knn_num*3
        heatmap_sum = torch.sum(heatmap_select, 2).view(batch_size, joint_num, 1, 1)  # B*self.joint_num*1
        heatmap_sum = heatmap_sum.repeat(1, 1, self.knn_num, 3)  # B*self.joint_num*knn_num*3
        point_select = points[unit_batch_index, unit_xyz_index, point_index.unsqueeze(-1).repeat(1, 1, 1, 3)]  # B*(self.joint_num)*(25*3)
        xyz = (vec_predict + point_select) * heatmap_select.unsqueeze(-1).repeat(1, 1, 1,3) / heatmap_sum  # B*self.joint_num*(25*3)
        xyz = (torch.sum(xyz.view(-1, joint_num, self.knn_num, 3), 2))  # B*self.joint_num*3
        return xyz
    # depth[-1,1]
    def depth2map(self, depth,heatmap_size=32):
        batchsize, jointnum = depth.size()
        depthmap = ((depth + 1) / 2).contiguous().view(batchsize, jointnum, 1, 1).expand(batchsize, jointnum, heatmap_size, heatmap_size)
        return depthmap

    #Geometric descriptor
    def joint2GD(self, joint):
        batchsize = joint.size()[0]
        _, jointnum, _ = joint.view(batchsize, -1, 3).size()
        project_joint = joint.view(batchsize, jointnum, 3).permute(0, 2, 1).view(batchsize, 3, jointnum, 1).expand(batchsize, 3, jointnum, jointnum)
        project_joint_t = project_joint.permute(0, 1, 3, 2)
        joint_diff = project_joint - project_joint_t
        joint_diff_square = joint_diff*joint_diff
        return torch.cat((joint_diff,joint_diff_square),dim=1)

    def img2mask(self, img):
        mask = torch.zeros_like(img)
        mask[torch.abs(img) < 0.95] = 1.0
        mask[torch.abs(img) > 0.95] = 0.0
        return mask

    def img2edt2(self, image_crop, pose3d, hmap_size):
        istep = 2. / hmap_size
        scale = int(image_crop.shape[0] / hmap_size)
        # image_hmap = image_crop
        if 1 == scale:
            image_hmap = (image_crop + 1) / 2
        else:
            image_hmap = (image_crop[::scale, ::scale] + 1) / 2  # downsampling
        mask = (image_hmap > 0.95)
        masked_edt = np.ma.masked_array(image_hmap, mask)
        edt_out = distance_transform_edt(mask) * istep
        edt = image_hmap - edt_out

        pose2d = pose3d[:, :2]
        pose2d = np.clip(np.floor((pose2d + 1) / 2 * hmap_size).astype(int),0,hmap_size-1)
        edt_l = []
        for pose in pose2d:
            val = edt[pose[1], pose[0]]
            if 0 > val:
                ring = np.ones_like(image_hmap)
                ring[pose[0], pose[1]] = 0
                ring = - distance_transform_edt(ring)
                ring = np.ma.masked_array(
                    ring, np.logical_and((val > ring), mask))
                ring = np.max(ring) - ring
                ring[~mask] = 0
                phi = image_hmap + ring + 1

            else:
                phi = masked_edt.copy()
            phi[pose[1], pose[0]] = 0.
            df = skfmm.distance(phi, dx=1e-1)

            df = np.ma.masked_array(df, mask)
            df_max = np.max(df)
            df = (df_max - df) / df_max
            df[mask] = 0.
            df[1. < df] = 0.  # outside isolated region have value > 1
            edt_l.append(df)
        return np.stack(edt_l, axis=2)

    def img2dist(self, image_crop, pose3d, hmap_size):

        scale = int(image_crop.shape[0] / hmap_size)
        if 1 == scale:
            image_hmap = image_crop
        else:
            image_hmap = image_crop[::scale, ::scale]

        xx, yy = np.meshgrid(  # xx: left --> right
            np.arange(hmap_size), np.arange(hmap_size))
        valid_id = np.where(image_hmap < 0.99)
        xx = xx[valid_id].astype(float)
        yy = yy[valid_id].astype(float)
        depth = image_hmap[valid_id]
        coord = np.vstack((yy, xx)).T / hmap_size
        depth_raw = np.hstack(((coord * 2 - np.ones([2]))[:, ::-1], depth.reshape(-1, 1)))
        theta = 250.0 / hmap_size
        hmap_l = []
        for joint in pose3d:
            offset = joint - depth_raw  # offset in raw 3d
            dist = linalg.norm(offset, axis=1)  # offset norm
            valid_id = np.where(np.logical_and(
                1e-4 < dist,  # remove sigular point
                theta > dist  # limit support within theta
            ))
            dist = dist[valid_id]
            dist = (theta - dist) / theta

            coord_vaild = coord[valid_id]
            img = np.zeros((hmap_size, hmap_size))

            xx = np.floor(coord_vaild[:, 0] * hmap_size).astype(int)
            yy = np.floor(coord_vaild[:, 1] * hmap_size).astype(int)
            img[xx, yy] = dist
            hmap_l.append(img)
        return np.stack(hmap_l, axis=2)

    def imgs2edt2(self, image_crop, pose3d, hmap_size):
        batchszie = image_crop.size()[0]
        imgs_np = image_crop.squeeze().cpu().numpy()
        poses_np = pose3d.view(batchszie,-1,3).cpu().detach().numpy()
        edts = np.expand_dims(self.img2edt2(imgs_np[0],poses_np[0],hmap_size).transpose((2,0,1)),axis=0)
        for index in range(1,batchszie):
            edts = np.concatenate((edts,np.expand_dims(self.img2edt2(imgs_np[index],poses_np[index],hmap_size).transpose((2,0,1)),axis=0)))
        return torch.from_numpy(edts).float().cuda()

    def imgs2dist(self, image_crop, pose3d, hmap_size):
        batchszie = image_crop.size()[0]
        imgs_np = image_crop.squeeze().cpu().numpy()
        poses_np = pose3d.view(batchszie,-1,3).cpu().detach().numpy()
        dists = np.expand_dims(self.img2dist(imgs_np[0], poses_np[0],hmap_size).transpose((2,0,1)),axis=0)
        for index in range(1,imgs_np.shape[0]):
            dists = np.concatenate((dists, np.expand_dims(self.img2dist(imgs_np[index], poses_np[index], hmap_size).transpose((2, 0, 1)),axis=0)))
        return torch.from_numpy(dists).float().cuda()

    def imgs2dist_tensor(self, image_crop, pose3d, hmap_size):
        scale = int(image_crop.size(-1) / hmap_size)
        image = image_crop.clone().unsqueeze(1).view(self.batch_size,1,128,128).repeat(1, self.joint_num, 1, 1)
        if scale != 1:
            image = image[:, :, ::scale, ::scale] # downsampling
        image_deep = image.clone()
        image_coord_x = self.mesh_edt_x/self.edt_size * 2 - 1
        image_coord_y = self.mesh_edt_y/self.edt_size * 2 - 1
        image_coord = torch.cat((image_coord_x.unsqueeze(-1), image_coord_y.unsqueeze(-1), image_deep.unsqueeze(-1)), dim=-1)
        joint_coord = pose3d.view(self.batch_size, self.joint_num, 1, 1, 3).repeat(1, 1, self.edt_size, self.edt_size, 1)

        image_coord = torch.sqrt(torch.sum(torch.pow(image_coord - joint_coord, 2), dim=-1) + 1e-6)
        img_mask = torch.ones_like(image).cuda()
        img_mask[image.abs().eq(1)] = 0
        image_coord = image_coord * img_mask
        image_coord_max, _ = image_coord.view(self.batch_size, self.joint_num, -1).max(-1)
        image_coord_max = image_coord_max.view(self.batch_size, self.joint_num, 1, 1).repeat(1, 1, self.edt_size, self.edt_size)
        image_coord = (image_coord_max - image_coord)/image_coord_max
        image_coord = image_coord * img_mask
        return image_coord

    def imgs2edt2_tensor(self, image_crop, pose3d, hmap_size):
        istep = 2. / hmap_size
        scale = int(image_crop.size(-1) / hmap_size)
        image = image_crop.clone().unsqueeze(1).view(self.batch_size,1,128,128).repeat(1, self.joint_num, 1, 1)
        if 1 == scale:
            image = (image + 1) / 2
        else:
            image = (image[:, :, ::scale, ::scale] + 1) / 2  # downsampling
        image_edt = image.clone()
        image_edt[image == 1] = float('inf')
        # image_edt[image_edt != 0] = 1

        pose3d[:, :, 0] = (pose3d[:, :, 0] + 1) / 2 * self.edt_size
        pose3d[:, :, 1] = (pose3d[:, :, 1] + 1) / 2 * self.edt_size
        joint_pos_x = pose3d[:, :, 0].view(self.batch_size, self.joint_num, 1, 1).repeat(1, 1, self.edt_size, self.edt_size).float()
        joint_pos_y = pose3d[:, :, 1].view(self.batch_size, self.joint_num, 1, 1).repeat(1, 1, self.edt_size, self.edt_size).float()
        heatmap = 1 - torch.exp(-(torch.pow((self.mesh_edt_x - joint_pos_x) / 1.7, 2) + torch.pow((self.mesh_edt_y - joint_pos_y) / 1.7, 2)))

        pose3d_index = torch.clamp(torch.floor(pose3d), 0, hmap_size - 1).long()
        my_mesh = torch.zeros([self.batch_size, self.joint_num, self.edt_size, self.edt_size], dtype=torch.uint8).cuda()
        my_mesh[self.batch_index, self.joint_index, pose3d_index[:,:,1], pose3d_index[:,:,0]] = 1

        solution, _ = self.fmm.march(my_mesh, distance=heatmap, speed=image_edt, batch_size=np.inf)
        solution_max, _ = solution.contiguous().view(self.batch_size, self.joint_num, -1).max(-1)
        solution_max = solution_max.view(self.batch_size, self.joint_num, 1, 1).repeat(1,1,self.edt_size,self.edt_size)
        solution = (solution_max - solution)/solution_max
        solution[image == 1] = 0
        return solution

    def geodesic_distance_transform(self, m):
        mask = m.mask
        visit_mask = mask.copy()  # mask visited cells
        m = m.filled(np.inf)
        m[m != 0] = np.inf
        distance_increments = np.asarray([np.sqrt(2), 1., np.sqrt(2), 1., 1., np.sqrt(2), 1., np.sqrt(2)])
        connectivity = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (not (i == j == 0))]
        cc = np.unravel_index(m.argmin(), m.shape)  # current_cell
        while (~visit_mask).sum() > 0:
            neighbors = [tuple(e) for e in np.asarray(cc) - connectivity
                         if not visit_mask[tuple(e)]]
            tentative_distance = [distance_increments[i] for i, e in enumerate(np.asarray(cc) - connectivity)
                                  if not visit_mask[tuple(e)]]
            for i, e in enumerate(neighbors):
                d = tentative_distance[i] + m[cc]
                if d < m[e]:
                    m[e] = d
            visit_mask[cc] = True
            m_mask = np.ma.masked_array(m, visit_mask)
            cc = np.unravel_index(m_mask.argmin(), m.shape)
        return m

