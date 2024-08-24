import os
import copy
import cv2
from PIL import Image
import numpy as np
import os.path as osp
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from bevlab.transforms import revtrans_rotation, revtrans_translation, revtrans_scaling
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

def get_P_from_Rt(R, t):
    P = np.identity(4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t
    return P

class NuscenesDataset(Dataset):
    CUSTOM_SPLIT = [
        "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
        "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
        "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
        "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
        "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
        "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
        "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
        "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
        "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
        "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
        "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
        "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
        "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
        "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
        "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
        "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
        "scene-1058", "scene-1094", "scene-1098", "scene-1107",
    ]

    def __init__(
        self,
        phase,
        config,
        **kwargs,
    ):
        self.phase = phase
        self.dataset_root = config.DATASET.DATASET_ROOT
        self.data_root = osp.join(self.dataset_root, 'data')
        # self.num_frames_in = config.DATASET.INPUT_FRAMES
        # self.num_frames_out = config.DATASET.OUTPUT_FRAMES
        # self.num_frames = self.num_frames_in + self.num_frames_out
        # self.select = config.DATASET.SKIP_FRAMES + 1
        self.voxel_size = config.DATASET.VOXEL_SIZE
        self.apply_scaling = config.DATASET.APPLY_SCALING
        self.img_h = config.DATASET.IMG_H
        self.img_w = config.DATASET.IMG_W
        self.flip_image_prob = 0.5

        # adhoc
        self.camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]

        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        elif config.DEBUG:
            self.nusc = NuScenes(
                version="v1.0-mini", dataroot=self.dataset_root, verbose=False
            )
        else:
            self.nusc = NuScenes(
                version="v1.0-trainval", dataroot=self.dataset_root, verbose=False
            )

        self.frame_list = list()
        # a skip ratio can be used to reduce the dataset size and accelerate experiments
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(self.CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = self.CUSTOM_SPLIT
        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                '''
                current_sample_token = scene["first_sample_token"]
                # Loop to get all successive keyframes
                sequence = []
                while current_sample_token != "":
                    current_sample = self.nusc.get("sample", current_sample_token)
                    sequence.append(current_sample)
                    current_sample_token = current_sample["next"]

                # Add new scans in the list
                for i in range(len(sequence)):
                    self.frame_list.append([sequence[j] for j in range(i, i + self.num_frames * self.select, self.select)])
                '''
                current_sample_token = scene["first_sample_token"]
                # Loop to get all successive keyframes
                while current_sample_token != "":
                    current_sample = self.nusc.get("sample", current_sample_token)
                    self.frame_list.append(current_sample)    
                    current_sample_token = current_sample["next"]        

    def get_sample_data_ego_pose_P(self, sample_data):
        sample_data_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
        sample_data_pose_R = np.asarray(Quaternion(sample_data_pose['rotation']).rotation_matrix).astype(np.float32)
        sample_data_pose_t = np.asarray(sample_data_pose['translation']).astype(np.float32)
        sample_data_pose_P = get_P_from_Rt(sample_data_pose_R, sample_data_pose_t)
        return sample_data_pose_P


    def get_calibration_P(self, sample_data):
        calib = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        R = np.asarray(Quaternion(calib['rotation']).rotation_matrix).astype(np.float32)
        t = np.asarray(calib['translation']).astype(np.float32)
        P = get_P_from_Rt(R, t)
        return P

    def get_camera_K(self, camera):
        calib = self.nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
        return np.asarray(calib['camera_intrinsic']).astype(np.float32)

    def load_point_cloud(self, sample):
        pointsensor = self.nusc.get("sample_data", sample["LIDAR_TOP"])
        pcl_path = osp.join(self.nusc.dataroot, pointsensor["filename"])
        points = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)[:, :4]
        points[:, 3] = points[:, 3] / 255
        lidar_calib_P = self.get_calibration_P(pointsensor)
        lidar_pose_P = self.get_sample_data_ego_pose_P(pointsensor)

        img_list = []
        R_list = []
        T_list = []
        K_list = []

        for camera_name in self.camera_list:
            camera = self.nusc.get('sample_data', sample[camera_name])
            img = np.array(Image.open(os.path.join(self.nusc.dataroot, camera['filename'])))

            K = self.get_camera_K(camera)

            camera_calib_P = self.get_calibration_P(camera)
            camera_pose_P = self.get_sample_data_ego_pose_P(camera)
            camera_pose_P_inv = np.linalg.inv(camera_pose_P)
            camera_calib_P_inv = np.linalg.inv(camera_calib_P)
            P_cam_pc = np.dot(camera_calib_P_inv, np.dot(camera_pose_P_inv,
                                                        np.dot(lidar_pose_P, lidar_calib_P)))
            R = P_cam_pc[0:3, 0:3]
            T = P_cam_pc[0:3, 3]
            img_list.append(img)
            R_list.append(R)
            T_list.append(T)
            K_list.append(K)

        return points, img_list, R_list, T_list, K_list

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        return_dict = dict()

        # step0: load data
        pc, img_list, R_list, T_list, K_list = self.load_point_cloud(self.frame_list[idx]['data'])

        # step 0.5: pc augmentation
        trans_dict = {}
        pc, trans_dict = revtrans_translation(pc, trans_dict)
        pc, trans_dict = revtrans_rotation(pc, trans_dict)

        matching_points_list = []
        matching_pixels_list = []
        pc_overlap_mask_list = []
        img_overlap_mask_list = []
        for cam_idx in range(len(self.camera_list)):
            img, R, T, K = img_list[cam_idx], R_list[cam_idx], T_list[cam_idx], K_list[cam_idx]

            # step1: resize image
            img_h_ori, img_w_ori, _ = img.shape
            img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32)
            img /= 255.0

            downsample_scale = 0.25
            scale_h = int(downsample_scale * self.img_h)
            scale_w = int(downsample_scale * self.img_w)
            img_h_ratio = scale_h / img_h_ori
            img_w_ratio = scale_w / img_w_ori

            K[0, 0] *= img_w_ratio
            K[0, 2] *= img_w_ratio
            K[1, 1] *= img_h_ratio
            K[1, 2] *= img_h_ratio

            # step1: image augmentation
            flip_image_mask=False
            if np.random.random() < self.flip_image_prob:
                np.fliplr(img)
                K[0, 2] = scale_w - K[0, 2]

            # step2: corresponding to pc augmentation
            T -= R @ trans_dict['T']
            R = R @ trans_dict['R'].T
            # if self.apply_scaling:
            #     pc, trans_dict = revtrans_scaling(pc, trans_dict)
            #     R = R * trans_dict['S']

            # step2: generate labels 
            ### label of overlap area ###
            cam = K @ (R @ pc[:, :3].T + T.reshape(3, 1))
            scan_C2 = cam.T
            scan_C2_depth = scan_C2[:, 2]
            scan_C2 = (scan_C2[:, :2].T / scan_C2[:, 2]).T

            inds = scan_C2[:, 0] >= 0
            inds = np.logical_and(inds, scan_C2[:, 0] <= scale_w-1)
            inds = np.logical_and(inds, scan_C2[:, 1] >= 0)
            inds = np.logical_and(inds, scan_C2[:, 1] <= scale_h-1)
            inds = np.logical_and(inds, scan_C2_depth >= 0)
            pc_overlap_mask = inds

            pixel_coordinates = np.round(scan_C2).astype(int)
            valid_indices = (pixel_coordinates[:, 0] >= 0) & (pixel_coordinates[:, 0] < scale_w) & \
                            (pixel_coordinates[:, 1] >= 0) & (pixel_coordinates[:, 1] < scale_h) & (scan_C2_depth > 0)
            pixel_coordinates = pixel_coordinates[valid_indices]
            img_overlap_mask = np.zeros((scale_h, scale_w))
            img_overlap_mask[pixel_coordinates[:, 1], pixel_coordinates[:, 0]] = 1 # shape=(H, W)

            matching_points = np.where(pc_overlap_mask)[0]
            matching_pixels = np.round(np.flip(scan_C2[matching_points], axis=1)).astype(np.int64)
            matching_pixels = np.concatenate((
                                    np.ones((matching_pixels.shape[0], 1), dtype=np.int64) * cam_idx,
                                    matching_pixels,
                                    ),axis=1)

            img_list[cam_idx] = img
            R_list[cam_idx] = R
            T_list[cam_idx] = T
            K_list[cam_idx] = K
            matching_points_list.append(matching_points)
            matching_pixels_list.append(matching_pixels)
            pc_overlap_mask_list.append(pc_overlap_mask)
            img_overlap_mask_list.append(img_overlap_mask)

            # # import pdb;pdb.set_trace()
            # return_dict["images"] = img
            # return_dict["points"] = pc
            # return_dict["cam_coords"] = np.concatenate([scan_C2, scan_C2_depth.reshape(-1, 1)], axis=1)
            # return_dict["R"] = R
            # return_dict["T"] = T
            # return_dict["pc_overlap_mask"] = pc_overlap_mask.astype(np.int)
            # return_dict["img_overlap_mask"] = img_overlap_mask

            if 0:
                import matplotlib.pyplot as plt
                # validate the correspondence between image and pc
                cam = K @ (R @ pc[:, :3].T + T.reshape(3, 1))
                scan_C2 = cam.T
                scan_C2_depth = scan_C2[:, 2]
                scan_C2 = (scan_C2[:, :2].T / scan_C2[:, 2]).T

                inds = scan_C2[:, 0] > 0
                inds = np.logical_and(inds, scan_C2[:, 0] < scale_w)
                inds = np.logical_and(inds, scan_C2[:, 1] > 0)
                inds = np.logical_and(inds, scan_C2[:, 1] < scale_h)
                inds = np.logical_and(inds, scan_C2_depth > 0)

                scale_img = cv2.resize(img, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
                plt.imshow(scale_img)
                plt.scatter(scan_C2[inds, 0], scan_C2[inds, 1], c=-scan_C2_depth[inds], alpha=0.5, s=0.2, cmap='viridis')
                # idx = np.random.choice(inds.sum(0), 1024, replace=False)
                # plt.scatter(scan_C2[inds, 0][idx], scan_C2[inds, 1][idx], c=-scan_C2_depth[inds][idx], alpha=0.5, s=0.2, cmap='viridis')

                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'tmp.jpg', bbox_inches='tight')
                #   plt.show()
                plt.close()

        img = np.array(img_list)
        img = img.transpose((0, 3, 1, 2))
        R_data = np.array(R_list)
        T_data = np.array(T_list)
        K_data = np.array(K_list)
        img_overlap_masks = np.array(img_overlap_mask_list)
        pairing_points = np.concatenate(matching_points_list)
        pairing_images = np.concatenate(matching_pixels_list)

        return_dict["images"] = img # (can, 3, h, w)
        return_dict["points"] = pc
        # return_dict["cam_coords"] = np.concatenate([scan_C2, scan_C2_depth.reshape(-1, 1)], axis=1)
        return_dict["R_data"] = R_data
        return_dict["T_data"] = T_data
        return_dict["K_data"] = K_data
        return_dict["img_overlap_masks"] = img_overlap_masks # (n, h, w)
        return_dict["pairing_points"] = pairing_points
        return_dict["pairing_images"] = pairing_images

        return return_dict


class SemanticKITTIDataset(Dataset):
    TRAIN_SET = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10}
    VALIDATION_SET = {8}
    TEST_SET = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

    def __init__(
        self,
        phase,
        config,
        **kwargs,
    ):

        if phase in ("val", "validation", "verifying"):
            phase_set = self.VALIDATION_SET
        else:
            phase_set = self.TRAIN_SET
        self.dataset_root = config.DATASET.DATASET_ROOT
        self.data_root = osp.join(self.dataset_root, 'dataset/sequences')
        # self.num_frames_in = config.DATASET.INPUT_FRAMES
        # self.num_frames_out = config.DATASET.OUTPUT_FRAMES
        # self.num_frames = self.num_frames_in + self.num_frames_out
        # self.select = config.DATASET.SKIP_FRAMES + 1
        self.voxel_size = config.DATASET.VOXEL_SIZE
        # self.bev_stride = config.OPTIMIZATION.BEV_STRIDE
        self.apply_scaling = config.DATASET.APPLY_SCALING
        self.img_h = config.DATASET.IMG_H
        self.img_w = config.DATASET.IMG_W

        self.frame_list = list()
        self.calib_tr_list = []
        self.calib_p2_list = []
        self.calib_p3_list = []
        for seq_n, num in enumerate(phase_set):
            directory = next(
                os.walk(
                    f"{self.data_root}/{num:0>2d}/velodyne"
                )
            )
            directory_sorted = np.sort(directory[2])
            poses = np.loadtxt(f"{self.data_root}/{num:0>2d}/poses.txt", dtype=np.float32).reshape(-1, 3, 4)
            poses = np.pad(poses, ((0,0),(0,1),(0,0)))
            poses[:, 3, 3] = 1.
            with open(f"{self.data_root}/{num:0>2d}/calib.txt", "r") as calib_file:
                lines = calib_file.readlines()
                line = lines[-1]
                assert line.startswith("Tr"), f"There is an issue with calib.txt in scene {num}"
                content = line.strip().split(":")[1]
                values = [float(v) for v in content.strip().split()]
                Tr = np.zeros((4, 4), dtype=np.float32)
                Tr[0, 0:4] = values[0:4]
                Tr[1, 0:4] = values[4:8]
                Tr[2, 0:4] = values[8:12]
                Tr[3, 3] = 1.0
                # pose_inv = np.linalg.inv(pose)

                line = lines[2]
                assert line.startswith("P2"), f"There is an issue with calib.txt in scene {num}"
                mat = np.fromstring(line[4:], sep=' ').reshape((3, 4)).astype(np.float32)
                # P2 = np.identity(4)
                # P2[0:3, :] = mat
                P2 = mat # shape 3x4

                line = lines[3]
                assert line.startswith("P3"), f"There is an issue with calib.txt in scene {num}"
                mat = np.fromstring(line[4:], sep=' ').reshape((3, 4)).astype(np.float32)
                # P3 = np.identity(4)
                # P3[0:3, :] = mat
                P3 = mat # shape 3x4
                self.calib_tr_list.append(Tr)
                self.calib_p2_list.append(P2)
                self.calib_p3_list.append(P3)

            sequence = list(
                map(
                    lambda x: f"{self.data_root}/"
                    f"{num:0>2d}/velodyne/" + x,
                    directory_sorted,
                )
            )
            # for i in range(len(sequence) - self.num_frames * self.select + 1):
                # self.frame_list.append([(sequence[j], poses[j], seq_n) for j in range(i, i + self.num_frames * self.select, self.select)])
            for i in range(len(sequence)):
                self.frame_list.append([sequence[i], seq_n])

        self.flip_image_prob = 0.5
        # self.camera_list = ['image_2', 'image_3']
        self.camera_list = ['image_2']

    def load_point_cloud(self, pc_file):
        points = np.fromfile(pc_file, dtype=np.float32).reshape((-1, 4))
        return points

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        return_dict = dict()
        pc_file, seq_q = self.frame_list[idx]
        pc = self.load_point_cloud(pc_file)


        trans_dict = {}
        pc, trans_dict = revtrans_translation(pc, trans_dict)
        pc, trans_dict = revtrans_rotation(pc, trans_dict)

        img_list = []
        R_list = []
        T_list = []
        K_list = []
        matching_points_list = []
        matching_pixels_list = []
        pc_overlap_mask_list = []
        img_overlap_mask_list = []        
        for cam_idx in range(len(self.camera_list)):
            cam_key = self.camera_list[cam_idx]

            img_file = pc_file.replace('velodyne', cam_key).replace('.bin', '.png')
            img = cv2.imread(img_file)
            if cam_key == 'image_2':
                P = self.calib_p2_list[seq_q].copy()
            else:
                P = self.calib_p3_list[seq_q].copy()

            # step1: resize image
            img_h_ori, img_w_ori, _ = img.shape
            img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32)
            img /= 255.0

            downsample_scale = 0.25
            scale_h = int(downsample_scale * self.img_h)
            scale_w = int(downsample_scale * self.img_w)
            img_h_ratio = scale_h / img_h_ori
            img_w_ratio = scale_w / img_w_ori

            ## break down P
            K = P[0:3, 0:3]
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            tz = P[2, 3]
            tx = (P[0, 3] - cx * tz) / fx
            ty = (P[1, 3] - cy * tz) / fy

            K[0, 0] *= img_w_ratio
            K[0, 2] *= img_w_ratio
            K[1, 1] *= img_h_ratio
            K[1, 2] *= img_h_ratio

            # step1: image augmentation
            flip_image_mask=False
            if np.random.random() < self.flip_image_prob:
                np.fliplr(img)
                K[0, 2] = scale_w - K[0, 2]

            Tr = self.calib_tr_list[seq_q]        
            R = Tr[:3, :3]
            T = P[:3, 3]
            #### resize image and change K end ####

            tz = T[2]
            tx = (P[0, 3] - cx * tz) / fx
            ty = (P[1, 3] - cy * tz) / fy
            T = np.array([tx, ty, tz])

            T -= R @ trans_dict['T']
            R = R @ trans_dict['R'].T
            # if self.apply_scaling:
            #     pc, trans_dict = revtrans_scaling(pc, trans_dict)
            #     R = R * trans_dict['S']            

            cam = K @ (R @ pc[:, :3].T + T.reshape(3, 1))
            scan_C2 = cam.T
            scan_C2_depth = scan_C2[:, 2]
            scan_C2 = (scan_C2[:, :2].T / scan_C2[:, 2]).T

            inds = scan_C2[:, 0] >= 0
            inds = np.logical_and(inds, scan_C2[:, 0] <= scale_w-1)
            inds = np.logical_and(inds, scan_C2[:, 1] >= 0)
            inds = np.logical_and(inds, scan_C2[:, 1] <= scale_h-1)
            inds = np.logical_and(inds, scan_C2_depth >= 0)
            pc_overlap_mask = inds

            pixel_coordinates = np.round(scan_C2).astype(int)
            valid_indices = (pixel_coordinates[:, 0] >= 0) & (pixel_coordinates[:, 0] < scale_w) & \
                            (pixel_coordinates[:, 1] >= 0) & (pixel_coordinates[:, 1] < scale_h) & (scan_C2_depth > 0)
            pixel_coordinates = pixel_coordinates[valid_indices]
            img_overlap_mask = np.zeros((scale_h, scale_w))
            img_overlap_mask[pixel_coordinates[:, 1], pixel_coordinates[:, 0]] = 1 # shape=(H, W)

            matching_points = np.where(pc_overlap_mask)[0]
            matching_pixels = np.round(np.flip(scan_C2[matching_points], axis=1)).astype(np.int64)
            matching_pixels = np.concatenate((
                                    np.ones((matching_pixels.shape[0], 1), dtype=np.int64) * cam_idx,
                                    matching_pixels,
                                    ),axis=1)

            img_list.append(img)
            R_list.append(R)
            T_list.append(T)
            K_list.append(K)
            matching_points_list.append(matching_points)
            matching_pixels_list.append(matching_pixels)
            pc_overlap_mask_list.append(pc_overlap_mask)
            img_overlap_mask_list.append(img_overlap_mask)            

            if 0:
                import matplotlib.pyplot as plt
                # validate the correspondence between image and pc
                K @ (R @ pc[:, :3].T + T.reshape(3, 1))
                scan_C2 = cam.T
                scan_C2_depth = scan_C2[:, 2]
                scan_C2 = (scan_C2[:, :2].T / scan_C2[:, 2]).T

                inds = scan_C2[:, 0] > 0
                inds = np.logical_and(inds, scan_C2[:, 0] < scale_w)
                inds = np.logical_and(inds, scan_C2[:, 1] > 0)
                inds = np.logical_and(inds, scan_C2[:, 1] < scale_h)
                inds = np.logical_and(inds, scan_C2_depth > 0)

                # img[img_overlap_mask==1] = 0
                scale_img = cv2.resize(img, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
                plt.imshow(scale_img)
                plt.scatter(scan_C2[inds, 0], scan_C2[inds, 1], c=-scan_C2_depth[inds], alpha=0.5, s=0.2, cmap='viridis')

                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'tmp_{idx}.jpg', bbox_inches='tight')
                #   plt.show()
                plt.close()


        img = np.array(img_list)
        img = img.transpose((0, 3, 1, 2))
        R_data = np.array(R_list)
        T_data = np.array(T_list)
        K_data = np.array(K_list)
        img_overlap_masks = np.array(img_overlap_mask_list)
        pairing_points = np.concatenate(matching_points_list)
        pairing_images = np.concatenate(matching_pixels_list)

        return_dict["images"] = img # (can, 3, h, w)
        return_dict["points"] = pc
        return_dict["R_data"] = R_data
        return_dict["T_data"] = T_data
        return_dict["K_data"] = K_data
        return_dict["img_overlap_masks"] = img_overlap_masks # (n, h, w)
        return_dict["pairing_points"] = pairing_points
        return_dict["pairing_images"] = pairing_images
        return return_dict

if __name__ == '__main__':
    from utils.config import generate_config, log_config
    config = generate_config('cfgs/pretrain_ns_minkunet.yaml')
    phase='train'

    # dataset = SemanticKITTIDataset(
    dataset = NuscenesDataset(
        phase=phase,
        config=config
    )
    for i in range(0, 5000, 10):
    # for i in range(len(dataset)):
        data = dataset[i]
    # data = dataset[1000]
    # data = dataset[2000]
