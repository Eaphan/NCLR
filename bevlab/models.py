import torch
import numpy as np
import torch.nn as nn
from bevlab import backbones
import torch.nn.functional as F
from torch_scatter import scatter

# import imagenet
from bevlab.imagenet import ImageEncoder
from bevlab.pnp import efficient_pnp

def create_correspondence_matrix(cam_coords, H, W, dist_thres=1, neg_margin=3):
    """
    创建一个 correspondence matrix，表示 3D 点与像素坐标系中每个点的对应关系。

    :param cam_coords: (N1, 2) 形状的 NumPy 数组，包含 3D 点在像素坐标系下的坐标。
    :param H: 像素坐标系的高度。
    :param W: 像素坐标系的宽度。
    :param dist_thres: 对应关系的距离阈值。
    :return: (H*W, N1) 形状的 correspondence matrix。
    """

    # 将 NumPy 数组转换为 PyTorch 张量
    cam_coords_tensor = torch.tensor(cam_coords, dtype=torch.float32)

    # 创建一个 (H, W, 2) 形状的网格，包含每个像素位置的坐标
    y_coords, x_coords = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    img_coords = torch.stack([x_coords, y_coords], dim=-1)  # (H, W, 2)

    # 展平 img_coords 以便于计算
    img_coords_flatten = img_coords.reshape(-1, 2)  # (H*W, 2)

    # 计算每个 3D 点与每个像素点之间的距离
    distances = torch.cdist(img_coords_flatten, cam_coords_tensor, p=2)

    # 应用阈值以创建 correspondence matrix
    correspondence_matrix = (distances <= dist_thres).float()  # Transpose to match the required shape (H*W, N1)
    # correspondence_matrix[(distances > dist_thres) & (distances < neg_margin)] = -1 # ignore fore loss

    return correspondence_matrix   

class SelfAttention1(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.scale = torch.sqrt(torch.tensor(embed_size, dtype=torch.float32))

    def forward(self, q, k, v):
        # q: (bs, c, n1)
        # k: (bs, c, n2)
        # v: (bs, c, n2)
        
        Q = self.query(q)  # (bs, c, n1)
        K = self.key(k)    # (bs, c, n2)
        V = self.value(v)  # (bs, c, n2)
        
        # Scaled dot-product attention
        energy = torch.bmm(Q.permute(0, 2, 1), K) / self.scale  # (bs, n1, n2)
        attention = F.softmax(energy, dim=-1)  # (bs, n1, n2)
        output = torch.bmm(attention, V.permute(0, 2, 1))  # (bs, n1, c)
        
        return output.permute(0, 2, 1)  # (bs, c, n1)

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.scale = torch.sqrt(torch.tensor(embed_size, dtype=torch.float32))

    def forward(self, q, k, v):
        # q: (bs, c, n1)
        # k: (bs, c, n2)
        # v: (bs, c, n2)
        
        # Adjust the shape of q, k, and v to (bs, n, c)
        q = q.permute(0, 2, 1)  # (bs, n1, c)
        k = k.permute(0, 2, 1)  # (bs, n2, c)
        v = v.permute(0, 2, 1)  # (bs, n2, c)

        Q = self.query(q)  # (bs, n1, c)
        K = self.key(k)    # (bs, n2, c)
        V = self.value(v)  # (bs, n2, c)
        
        # Scaled dot-product attention
        energy = torch.bmm(Q, K.permute(0, 2, 1)) / self.scale  # (bs, n1, n2)
        attention = F.softmax(energy, dim=-1)  # (bs, n1, n2)
        output = torch.bmm(attention, V)  # (bs, n1, c)
        
        return output.permute(0, 2, 1)  # (bs, c, n1)


class BEVTrainer(nn.Module):  # TODO rename
    def __init__(self, config):
        super().__init__()
        encoder_class = getattr(backbones, config.ENCODER.NAME)
        self.range = config.DATASET.POINT_CLOUD_RANGE
        self.voxel_size = config.DATASET.VOXEL_SIZE
        self.scale = self.range[3]
        self.encoder = encoder_class(config.ENCODER.IN_CHANNELS, config.ENCODER.OUT_CHANNELS, config=config)

        # self.pc_feature_layer=nn.Sequential(
        #                         nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),
        #                         nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),
        #                         nn.Conv1d(128,64,1,bias=False))
        self.pc_feature_layer = nn.Identity()

        self.pc_conv_before_attn = nn.Conv1d(64, 64, 1, bias=False)
        # self.pc_score_layer=nn.Sequential(nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),
        #                         nn.Conv1d(128,64,1,bias=False),nn.BatchNorm1d(64),nn.ReLU(),
        #                         nn.Conv1d(64,1,1,bias=False),nn.Sigmoid())

        # self.img_encoder=imagenet.ImageEncoder()
        self.img_encoder=ImageEncoder()
        # self.up_conv1=ImageUpSample(512+256,256)
        # self.up_conv2=ImageUpSample(256+128,128)
        # self.up_conv3=ImageUpSample(128+64+64,128)
        self.img_feature_layer = nn.Identity()
        self.img_conv_before_attn = nn.Conv2d(64, 64, 1, bias=False)

        self.i2p_fuse_layer = SelfAttention(64)
        self.p2i_fuse_layer = SelfAttention(64)

        self.pc_score_layer = nn.Sequential(nn.Conv1d(64,64,1,bias=False),nn.BatchNorm1d(64),nn.ReLU(),
                                            nn.Conv1d(64,64,1,bias=False),nn.BatchNorm1d(64),nn.ReLU(),
                                            nn.Conv1d(64, 1, 1, bias=False), nn.Sigmoid())
        self.img_score_layer = nn.Sequential(nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),
                                             nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),
                                             nn.Conv2d(64, 1, 1, bias=False), nn.Sigmoid())

        # self.img_feature_layer = ResidualConv(64,128,kernel_1=True)
        # self.img_feature_layer = nn.Sequential(nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),
        #                                      nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),
        #                                      nn.Conv2d(64,64,1,bias=False))

        # self.img_score_layer = nn.Sequential(nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),
        #                                     nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),
        #                                     nn.Conv2d(64,1,1,bias=False),nn.Sigmoid())                

        # self.input_frames = config.DATASET.INPUT_FRAMES
        # self.output_frames = config.DATASET.OUTPUT_FRAMES
        # self.total_frames = self.input_frames + self.output_frames
        # self.loss = config.OPTIMIZATION.LOSS
        # self.bev_stride = config.OPTIMIZATION.BEV_STRIDE
        self.batch_first = config.ENCODER.COLLATE == "collate_minkowski"
        self.collapse = config.ENCODER.COLLATE != "collate_spconv"

        self.kpt_pc_num = 1024 # per sample
        # self.loss_contras_func =MultiMatchInfoNCELoss(0.07)
        self.criterion = NCELoss(0.07)
        self.overlap_criterion = nn.BCELoss()


    def forward(self, batch):
        # ['points', 'R', 'T', 'pc_overlap_mask', 'img_overlap_mask', 'voxels', 'indexes', 'inv_indexes', 'batch_n_points', 'pc_min', 'coordinates']
        # batch_n_points是batch中每个元素对应的voxel数量，pc_min是中间变量原坐标除以voxel_size之后 最小值.
        # 是list类型的包括: points, pc_overlap_mask, img_overlap_mask, indexes, inv_indexes, batch_n_points, coordinates.
        # R: [2,3,3] T: [2, 3], voxels: [N, 4],
        batch_size = len(batch['batch_n_points'])
        device = batch['voxels'].device
        input_fmap = self.encoder(batch['voxels'], batch['coordinates']) # (n,c)
        voxel_feature = input_fmap # (1, c, n)
        voxel_feature_norm=F.normalize(voxel_feature, dim=1,p=2)
        voxel_feature_norm = voxel_feature_norm # (N, 64)

        # batch['images'] = batch['images'].permute(0, 3, 1, 2).contiguous()
        img_feature=self.img_encoder(batch['images']) # (bs, c, h, w)
        cam_num = img_feature.shape[0] // batch_size

        # img_score=self.img_score_layer(image_feature_mid)
        img_feature_norm=F.normalize(img_feature, dim=1,p=2) # (bs*N_cam, c, h, w)
        img_feature_h = img_feature.shape[2]
        img_feature_w = img_feature.shape[3]

        ################# feature fusion #################
        # 对于每个图像, sample 部分 voxel_feature 来和 图像特征融合;
        sample_voxel_num = 4096
        c_sample_voxel_feature_list = []
        c_sample_voxel_feature_norm_list = []
        c_sample_voxel_abs_coords_list = []
        c_sample_voxel_2d_coords_list = []
        c_sample_voxel_pairing_images_list = []
        sample_voxel_mask_list = []
        for batch_idx in range(batch_size):
            batch_indices = batch['coordinates'][:, 3] == batch_idx
            b_voxel_num = (batch_indices).sum()
            b_voxel_feature = voxel_feature[batch_indices]
            b_voxel_feature_norm = voxel_feature_norm[batch_indices]
            
            b_voxel_abs_coords = torch.tensor(batch['points'][batch_idx], device=img_feature.device)[batch['indexes'][batch_idx]]
            for cam_idx in range(cam_num):
                # 每个图像随机采样 sample_voxel_num 个点
                c_sample_indices = torch.randint(0, b_voxel_num, (sample_voxel_num,))
                # 获得每个batch_idx 对应的有效 voxel 数量，然后采样; sample_ids
                # 根据sample_ids 来获得点云特征，以及对应的pc的坐标
                c_sample_voxel_feature = b_voxel_feature[c_sample_indices]
                c_sample_voxel_feature_norm = b_voxel_feature_norm[c_sample_indices]
                c_sample_voxel_abs_coords = b_voxel_abs_coords[c_sample_indices, :3]
                # c_sample_voxel_abs_coords = torch.tensor(c_sample_voxel_abs_coords, device=img_feature.device)
                
                # cam = K @ (R @ pc[:, :3].T + T.reshape(3, 1))
                R = batch['R_data'][batch_idx][cam_idx]
                T = batch['T_data'][batch_idx][cam_idx]
                K = batch['K_data'][batch_idx][cam_idx]
                c_sample_voxel_2d_coords = K @ (R @ c_sample_voxel_abs_coords.T + T.reshape(3, 1))
                c_sample_voxel_2d_coords = c_sample_voxel_2d_coords.T
                c_sample_voxel_2d_coords = (c_sample_voxel_2d_coords[:, :2] / c_sample_voxel_2d_coords[:, 2:3])

                tmp_mask = torch.zeros(voxel_feature.shape[0], device=voxel_feature.device)
                cam_mask_all = batch['pairing_images'][:, 0] == (batch_idx * cam_num + cam_idx)
                cam_pairing_points = batch['pairing_points'][cam_mask_all]
                tmp_mask[cam_pairing_points] = 1
                cam_mask_sample = tmp_mask[batch_indices][c_sample_indices]

                tmp_images = torch.zeros((voxel_feature.shape[0], 3), device=img_feature.device, dtype=batch['pairing_images'].dtype)
                tmp_images[cam_pairing_points] = batch['pairing_images'][cam_mask_all]
                c_sample_voxel_pairing_images = tmp_images[batch_indices][c_sample_indices]                

                c_sample_voxel_feature_list.append(c_sample_voxel_feature)
                c_sample_voxel_feature_norm_list.append(c_sample_voxel_feature_norm)
                c_sample_voxel_abs_coords_list.append(c_sample_voxel_abs_coords)
                c_sample_voxel_2d_coords_list.append(c_sample_voxel_2d_coords)
                c_sample_voxel_pairing_images_list.append(c_sample_voxel_pairing_images)
                sample_voxel_mask_list.append(cam_mask_sample)

        sample_voxel_feature = torch.stack(c_sample_voxel_feature_list, 0).permute(0, 2, 1)
        sample_voxel_mask = torch.stack(sample_voxel_mask_list, 0)

        # img_feature
        sample_voxel_feature = self.pc_conv_before_attn(sample_voxel_feature)
        img_feature_mid = self.img_conv_before_attn(img_feature)
        img_feature_mid = img_feature_mid.view(batch_size * cam_num, -1, img_feature_h * img_feature_w)

        # fuse feature
        fused_pc_feature = sample_voxel_feature + self.i2p_fuse_layer(sample_voxel_feature, img_feature_mid, img_feature_mid)

        # (bs*cam_num, c, img_h, img_w)
        fused_img_feature = img_feature_mid + self.i2p_fuse_layer(img_feature_mid, sample_voxel_feature, sample_voxel_feature)
        fused_pc_feature_norm = F.normalize(fused_pc_feature, dim=1,p=2)
        fused_img_feature_norm = F.normalize(fused_img_feature, dim=1,p=2)

        ################# feature fusion ################# done
        # select points for contrastive learning with fused features
        fuse_contras_pc_feature_norm = []
        fuse_contras_img_feature_norm = []
        for idx in range(batch_size * cam_num):
            # 筛选出 在overlap区域的pair
            voxel_mask_iter = sample_voxel_mask[idx]
            fused_pc_feature_norm_sample_valid = fused_pc_feature_norm[idx].T[voxel_mask_iter==1]
            pairing_images_sample_valid = c_sample_voxel_pairing_images_list[idx][voxel_mask_iter==1, 1:]
            m = tuple(pairing_images_sample_valid.T.long())
            fused_img_feature_norm_sample_valid = fused_img_feature_norm[idx].view(64, img_feature_h, img_feature_w).permute(1, 2, 0)[m]
            fuse_contras_pc_feature_norm.append(fused_pc_feature_norm_sample_valid)
            fuse_contras_img_feature_norm.append(fused_img_feature_norm_sample_valid)
        fuse_contras_pc_feature_norm = torch.cat(fuse_contras_pc_feature_norm, 0)
        fuse_contras_img_feature_norm = torch.cat(fuse_contras_img_feature_norm, 0)
        choice_idx = np.random.choice(fuse_contras_pc_feature_norm.shape[0], self.kpt_pc_num * batch_size, replace=False)
        loss_contras_fuse = self.criterion(fuse_contras_pc_feature_norm[choice_idx], fuse_contras_img_feature_norm[choice_idx])


        # overlapping estimation score
        fused_img_feature = fused_img_feature.view(batch_size * cam_num, -1, img_feature_h, img_feature_w)
        pc_score = self.pc_score_layer(fused_pc_feature).squeeze() # (bs*cam_num, 1, sample_voxel_num) -> (bs*cam_num, sample_voxel_num)
        img_score = self.img_score_layer(fused_img_feature).squeeze() # (bs*cam_num, 1, img_feature_h, img_feature_w)

        ################# keypoint selection #################
        num_keypoints = 128
        keypoints_feature_list = []
        keypoints_coordinates_list = []
        keypoints_2d_coordinates_list = []
        pose_valid_mask = torch.ones((batch_size * cam_num,), dtype=torch.bool)

        pc_cls_precision_list = []
        # precision_th0.9_list = []
        # precision_th0.8_list = []
        for idx in range(batch_size * cam_num):
            pc_cls_precision_list.append((sample_voxel_mask[idx][pc_score[idx] > 0.7].sum() / (pc_score[idx] > 0.7).sum()).item())

            keypoints_mask_current = pc_score[idx] > 0.9
            # keypoints_mask_current = sample_voxel_mask[idx] > 0.8 # ad hoc for debug
            keypoints_current_valid_num = (keypoints_mask_current).sum()
            # print('# first keypoints_current_valid_num', keypoints_current_valid_num)

            valid_indices = torch.nonzero(keypoints_mask_current, as_tuple=False).squeeze()
            if keypoints_current_valid_num > num_keypoints:
                indices = valid_indices[torch.randperm(valid_indices.size(0))[:num_keypoints]]
            elif keypoints_current_valid_num > 10:
                # indices = torch.topk(sample_voxel_mask[idx], num_keypoints)[1]
                num_pad = num_keypoints - valid_indices.size(0)
                # 使用现有索引填充至目标数量
                pad_indices = valid_indices[torch.randint(0, valid_indices.size(0), (num_pad,))]
                indices = torch.cat((valid_indices, pad_indices), dim=0)
            else:
                keypoints_mask_current = pc_score[idx] > 0.7
                # keypoints_mask_current = sample_voxel_mask[idx] > 0.8 # ad hoc for debug
                keypoints_current_valid_num = (keypoints_mask_current).sum()
                # print('# second keypoints_current_valid_num', keypoints_current_valid_num)
                valid_indices = torch.nonzero(keypoints_mask_current, as_tuple=False).squeeze()
                if keypoints_current_valid_num > num_keypoints:
                    indices = valid_indices[torch.randperm(valid_indices.size(0))[:num_keypoints]]
                elif keypoints_current_valid_num > 10:
                    # indices = torch.topk(sample_voxel_mask[idx], num_keypoints)[1]
                    num_pad = num_keypoints - valid_indices.size(0)
                    # 使用现有索引填充至目标数量
                    pad_indices = valid_indices[torch.randint(0, valid_indices.size(0), (num_pad,))]
                    indices = torch.cat((valid_indices, pad_indices), dim=0)
                else:
                    pose_valid_mask[idx] = False
                    # topk of pc_score[idx]
                    valid_indices = torch.topk(pc_score[idx], 10)[1]
                    num_pad = num_keypoints - valid_indices.size(0)
                    pad_indices = valid_indices[torch.randint(0, valid_indices.size(0), (num_pad,))]
                    indices = torch.cat((valid_indices, pad_indices), dim=0)

            keypoints_coordinates_current = c_sample_voxel_abs_coords_list[idx][indices]
            keypoints_2d_coordinates_current = c_sample_voxel_2d_coords_list[idx][indices]
            # keypoints_features_current = c_sample_voxel_feature_norm_list[idx][indices]
            keypoints_features_current = fused_pc_feature_norm[idx].T[indices]

            keypoints_feature_list.append(keypoints_features_current)
            keypoints_coordinates_list.append(keypoints_coordinates_current)
            keypoints_2d_coordinates_list.append(keypoints_2d_coordinates_current)

        keypoints_features = torch.stack(keypoints_feature_list, 0) # (bs*cam_num, k, 64)
        keypoints_coordinates = torch.stack(keypoints_coordinates_list, 0) # (bs*cam_num, k, 3)
        keypoints_2d_coordinates = torch.stack(keypoints_2d_coordinates_list, 0) # (bs*cam_num, k, 2)

        # img_feature_norm_flatten = img_feature_norm.view(batch_size * cam_num, -1, img_feature_h * img_feature_w)
        # mutual_info = torch.bmm(keypoints_features, img_feature_norm_flatten) / 0.01
        
        mutual_info = torch.bmm(keypoints_features, fused_img_feature_norm) / 0.01

        # 生成x坐标和y坐标的网格
        x_coords = torch.linspace(0, img_feature_w - 1, img_feature_w, device=img_feature.device)
        y_coords = torch.linspace(0, img_feature_h - 1, img_feature_h, device=img_feature.device)
        # 使用 meshgrid 生成二维网格
        img_x, img_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        # 扩展并调整形状
        img_x = img_x.T.unsqueeze(0).unsqueeze(1).expand(img_feature.size(0), -1, -1, -1)
        img_y = img_y.T.unsqueeze(0).unsqueeze(1).expand(img_feature.size(0), -1, -1, -1)
        # 合并 x 和 y 坐标
        img_xy = torch.cat((img_x, img_y), dim=1)
        # 将坐标展平
        img_xy_flatten = img_xy.view(img_feature.size(0), 2, -1)

        soft_matching_matrix = F.softmax(mutual_info, dim=2) # [batch_size, num_keypoints, height*width]
        predicted_2d_coords = torch.bmm(soft_matching_matrix, img_xy_flatten.permute(0, 2, 1)) # [batch_size, num_keypoints, 2]
        
        # max_indices = torch.argmax(mutual_info, dim=2)  # [batch_size, num_keypoints]
        # batch_indices = torch.arange(mutual_info.size(0)).unsqueeze(1).expand_as(max_indices)  # [batch_size, num_keypoints]
        # predicted_2d_coords_x = img_xy_flatten[:, 0, :].gather(1, max_indices)  # [batch_size, num_keypoints]
        # predicted_2d_coords_y = img_xy_flatten[:, 1, :].gather(1, max_indices)  # [batch_size, num_keypoints]
        # predicted_2d_coords = torch.stack((predicted_2d_coords_x, predicted_2d_coords_y), dim=2) 

        # print(predicted_2d_coords[0][:7])
        # print(keypoints_2d_coordinates[0][:7])

        y_homo = torch.cat([predicted_2d_coords, torch.ones(batch_size * cam_num, num_keypoints, 1, device=predicted_2d_coords.device)],dim=-1)
        K_data = batch['K_data'].view(batch_size * cam_num, 3, 3)
        K_inv = torch.inverse(K_data).permute(0, 2, 1)
        y_uncalibrated = torch.bmm(y_homo, K_inv)
        y_uncalibrated = y_uncalibrated[:, :, :2]
        x_cam, R, t, err_2d, err_3d = efficient_pnp(keypoints_coordinates, y_uncalibrated)

        # y_homo = torch.cat([keypoints_2d_coordinates, torch.ones(batch_size * cam_num, num_keypoints, 1, device=keypoints_2d_coordinates.device)],dim=-1)
        # y_uncalibrated = torch.bmm(y_homo, K_inv)
        # y_uncalibrated = y_uncalibrated[:, :, :2]
        # x_cam, R, t, err_2d, err_3d = efficient_pnp(keypoints_coordinates, y_uncalibrated)


        def smooth_l1_loss(input, target, beta=1.0):
            diff = input - target
            abs_diff = diff.abs()
            loss = torch.where(abs_diff < beta, 0.5 * diff ** 2 / beta, abs_diff - 0.5 * beta)
            return loss.mean()
        R_gt = batch['R_data'].view(batch_size * cam_num, 3, 3)
        t_gt = batch['T_data'].view(batch_size * cam_num, 3)
        R = R.permute(0, 2, 1)
        loss_pose = smooth_l1_loss(R[pose_valid_mask], R_gt[pose_valid_mask]) + 0.01 * smooth_l1_loss(t[pose_valid_mask], t_gt[pose_valid_mask])
        # loss_pose = smooth_l1_loss(R, R_gt) + smooth_l1_loss(t, t_gt)

        # overlapping loss
        loss_overlap_pc = self.overlap_criterion(pc_score, sample_voxel_mask)
        loss_overlap_img = self.overlap_criterion(img_score, batch['img_overlap_masks'])

        ################# contrastive loss #################
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        idx = np.random.choice(pairing_points.shape[0], self.kpt_pc_num * batch_size, replace=False)
        match_voxel_feature = voxel_feature_norm[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        match_img_feature = img_feature_norm.permute(0, 2, 3, 1)[m]
        loss_contras = self.criterion(match_voxel_feature, match_img_feature)
        ################# contrastive loss ################# done

        metrics = {'loss_contras': loss_contras.item(), 'loss_contras_fuse': loss_contras_fuse.item(),
                 'loss_overlap_pc': loss_overlap_pc.item(), 'loss_overlap_img': loss_overlap_img.item(),
                 'pc_cls_precision': np.mean(np.array(pc_cls_precision_list)[pose_valid_mask.numpy()]),
                 'loss_pose': loss_pose.item()}
        
        # Note: enable loss_pose after specific epochs
        # ad hoc epochs, try to tune the settings to reach best performances
        if batch['cur_epoch'] >= 45:
            loss = loss_contras + loss_contras_fuse + loss_overlap_pc + loss_overlap_img + loss_pose 
            # try:
            # metrics.update({'loss_pose': loss_pose.item()})
            # except:
        else:
            loss = loss_contras + loss_contras_fuse + loss_overlap_pc + loss_overlap_img

        return loss, metrics


class MultiMatchInfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(MultiMatchInfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, q, k, m):
        """
        q: Tensor of shape (n1, c) representing image pixel features.
        k: Tensor of shape (n2, c) representing point cloud features.
        m: Binary match matrix of shape (n1, n2) where m[i, j] = 1 indicates a match.
        """

        # Compute the dot product between q and k (logits)
        logits = torch.mm(q, k.transpose(1, 0)) / self.temperature

        # Compute the softmax over the logits
        softmaxed_logits = F.softmax(logits, dim=1)
        # Create a mask for the positive samples
        positive_mask = m
        # Calculate the loss for the positive samples
        positive_logits = torch.log(softmaxed_logits + 1e-8) * positive_mask
        positive_loss_row = -positive_logits.sum(dim=1)

        softmaxed_logits = F.softmax(logits, dim=0)
        positive_logits = torch.log(softmaxed_logits + 1e-8) * positive_mask
        positive_loss_col = -positive_logits.sum(dim=0)
        
        # Since every row in q should match with one or more columns in k, 
        # we can simply average the positive loss over all rows in q.
        # print('###', positive_loss_row.max(), positive_loss_col.max())
        # print('###', positive_loss_row.shape, positive_loss_col.shape, positive_loss_row.max(), positive_loss_col.max())
        loss = positive_loss_row.mean() + positive_loss_col.mean()
        return loss


class MultiMatchInfoNCELoss_new(nn.Module):
    def __init__(self, temperature=1.0):
        super(MultiMatchInfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, q, k, m):
        """
        q: Tensor of shape (n1, c) representing image pixel features.
        k: Tensor of shape (n2, c) representing point cloud features.
        m: Binary match matrix of shape (n1, n2) where m[i, j] = 1 indicates a match, -1 to ignore.
        """

        # Compute the dot product between q and k (logits)
        logits = torch.mm(q, k.transpose(1, 0)) / self.temperature

        # Create masks for valid (1 or 0) and positive (1) samples
        valid_mask = m >= 0
        positive_mask = m > 0

        # Mask the logits before softmax, setting ignored entries to a very large negative value
        logits_masked = logits.masked_fill(~valid_mask, -1e9)

        # Compute softmax over the masked logits for both directions
        softmaxed_logits_row = F.softmax(logits_masked, dim=1)
        softmaxed_logits_col = F.softmax(logits_masked, dim=0)

        # Calculate the loss for positive samples
        positive_loss_row = -torch.log(softmaxed_logits_row + 1e-8) * positive_mask
        positive_loss_col = -torch.log(softmaxed_logits_col + 1e-8) * positive_mask

        # Calculate mean loss
        # loss_row = positive_loss_row.sum(dim=1) / valid_mask.sum(dim=1)
        # loss_col = positive_loss_col.sum(dim=0) / valid_mask.sum(dim=0)
        # # Handle possible division by zero
        # loss_row[torch.isnan(loss_row)] = 0
        # loss_col[torch.isnan(loss_col)] = 0
        loss_row = positive_loss_row.sum(dim=1)
        loss_col = positive_loss_col.sum(dim=0)


        # Final loss is the mean of row-wise and column-wise losses
        loss = loss_row.mean() + loss_col.mean()
        return loss


class MultiMatchInfoNCELoss_v3(nn.Module):
    # softmax -> sigmoid
    def __init__(self, temperature=1.0):
        super(MultiMatchInfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, q, k, m):
        """
        q: Tensor of shape (n1, c) representing image pixel features.
        k: Tensor of shape (n2, c) representing point cloud features.
        m: Binary match matrix of shape (n1, n2) where m[i, j] = 1 indicates a match, 0 or -1 to ignore.
        """

        # Compute the dot product between q and k (logits)
        logits = torch.mm(q, k.transpose(1, 0)) / self.temperature

        # Apply sigmoid to logits
        sigmoid_logits = torch.sigmoid(logits)

        # Create masks for valid (1 or 0) and positive (1) samples
        valid_mask = m >= 0
        positive_mask = m > 0

        # Calculate the binary cross-entropy loss
        # Target is 1 for positive samples and 0 for others
        targets = positive_mask.float()
        bce_loss = F.binary_cross_entropy(sigmoid_logits, targets, reduction='none')

        # Apply the valid mask, ignoring -1 entries
        bce_loss = bce_loss * valid_mask.float()

        # Calculate the mean loss
        loss = bce_loss.sum() / valid_mask.sum()

        return loss

def make_models(config):
    model = BEVTrainer(config)
    return model


class NCELoss(nn.Module):
    """
    Compute the PointInfoNCE loss
    """

    def __init__(self, temperature):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k, q):
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.temperature)
        out = out.contiguous()

        loss = self.criterion(out, target)
        return loss
