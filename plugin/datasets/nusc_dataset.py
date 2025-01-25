from.base_dataset import BaseMapDataset
from .map_utils.nuscmap_extractor import NuscMapExtractor
from mmdet.datasets import DATASETS
import numpy as np
from .visualize.renderer import Renderer
import mmcv
from time import time
from pyquaternion import Quaternion

from shapely.geometry import LineString, Polygon

from nuscenes.eval.common.utils import quaternion_yaw
import math
import torch 


@DATASETS.register_module()
class NuscDataset(BaseMapDataset):
    """NuScenes map dataset class.

    Args:
        ann_file (str): annotation file path
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        eval_config (Config): evaluation config
        meta (dict): meta information
        pipeline (Config): data processing pipeline config
        interval (int): annotation load interval
        work_dir (str): path to work dir
        test_mode (bool): whether in test mode
    """
    
    def __init__(self, data_root, **kwargs):
        super().__init__(**kwargs)
        
        self.data_root = data_root
        self.map_extractor = NuscMapExtractor(data_root, self.roi_size)
        self.renderer = Renderer(self.cat2id, self.roi_size, 'nusc')
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.

        """

        if self.maptr_v2:
            
            data = mmcv.load(ann_file) #From MaptrV2
    
            data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
            data_infos = data_infos[::self.interval]
            self.metadata = data['metadata']
            self.version = self.metadata['version']
            self.samples = data_infos
            print("Ann collected!")

        else:

            start_time = time()
            ann = mmcv.load(ann_file)
            samples = list(ann)[::self.interval]
            
            print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
            self.samples = samples

  



    def get_sample(self, idx):
        """Get data sample. For each sample, map extractor will be applied to extract 
        map elements. 

        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """

        sample = self.samples[idx]
        location = sample['location']
        #         location
        # 'boston-seaport'

        #     sample['e2g_translation']
        # [332.0797577474775, 659.4188573738066, 0.0]

        # sample['e2g_rotation']
        # [-0.14457762634660634, 0.007687923603545458, 0.001974063238896858, -0.9894616257667479]   
             
        map_geoms = self.map_extractor.get_map_geom(location, sample['e2g_translation'], 
                sample['e2g_rotation'])

        # count = {'divider': [0,0], 'ped_crossing': [0,0], 'boundary': [0,0], 'drivable_area': [0,0]}
        map_label2geom = {}
        for k, v in map_geoms.items(): # divider line string, ped cros line string, driv are polygon
            # lines = sum([1 if o.geom_type=='LineString' else 0 for o in v]) + count[k][0]
            # poly = sum([1 if o.geom_type=='Polygon' else 0 for o in v]) + count[k][1]

            # count[k] = [lines, poly]
            if k in self.cat2id.keys():
                map_label2geom[self.cat2id[k]] = v
        
        ego2img_rts = []
        for c in sample['cams'].values():
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])
            ego2cam_rt = extrinsic # ego -> cam 
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt) # ego -> cam_rt  -> img
            ego2img_rts.append(ego2cam_rt)

        # if sample['sample_idx'] == 0:
        #     is_first_frame = True
        # else:
        #     is_first_frame = self.flag[sample['sample_idx']] > self.flag[sample['sample_idx'] - 1]
        input_dict = {
            'location': location,
            'token': sample['token'],
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            # intrinsics are 3x3 Ks
            'cam_intrinsics': [c['intrinsics'] for c in sample['cams'].values()],
            # extrinsics are 4x4 tranform matrix, **ego2cam**
            'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'map_geoms': map_label2geom, # {0: List[ped_crossing(LineString)], 1: ...}
            'ego2global_translation': sample['e2g_translation'], 
            'ego2global_rotation': Quaternion(sample['e2g_rotation']).rotation_matrix.tolist(),
            # 'is_first_frame': is_first_frame, # deprecated
            'sample_idx': sample['sample_idx'],
            'scene_name': sample['scene_name']
            # 'group_idx': self.flag[sample['sample_idx']]
        }

        return input_dict
    
    def get_data_info(self, idx):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.samples[idx]
        # standard protocal modified from SECOND.Pytorch
        location = info['map_location']

    
        # input_dict['sweeps'][0].keys()
        # dict_keys(['data_path', 'type', 'sample_data_token', 'sensor2ego_translation', 'sensor2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'sensor2lidar_rotation', 'sensor2lidar_translation'])

        # input_dict['pts_filename']
        # './data/nuscenes/samples/LIDAR_TOP/n015-2018-09-27-15-33-17+0800__LIDAR_TOP__1538033900198208.pcd.bin'

        # input_dict['can_bus'].shape (18,)
        # lidar to ego transform
        
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]

        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        camera2ego = []
        cam_intrinsics = []
        cam_extrinsics = []

        ego2img_rts = []

        
        camego2global_list = []
        for cam_type, cam_info in info['cams'].items():


            new_path = self.data_root + cam_info['data_path'].split('nuscenes')[-1]
            image_paths.append(new_path)

            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            lidar2cam_rt_t = lidar2cam_rt.T

            # if self.noise == 'rotation':
            #     lidar2cam_rt_t = add_rotation_noise(lidar2cam_rt_t, std=self.noise_std)
            # elif self.noise == 'translation':
            #     lidar2cam_rt_t = add_translation_noise(
            #         lidar2cam_rt_t, std=self.noise_std)



            intrinsic = cam_info['cam_intrinsic']
        
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt_t) # img -> cam
            lidar2img_rts.append(lidar2img_rt)

            cam_intrinsics.append(intrinsic)
            lidar2cam_rts.append(lidar2cam_rt_t)

    

            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                cam_info["sensor2ego_rotation"]
            ).rotation_matrix 
            camera2ego[:3, 3] = cam_info["sensor2ego_translation"]
            #camera2ego.append(camera2ego) # float32¨ # cam -> ego 

            extrinsic = np.linalg.inv(camera2ego)
            cam_extrinsics.append(extrinsic)

            ego2cam_rt = (viewpad @ extrinsic)
            ego2img_rts.append(ego2cam_rt)

            
            # # camego to global transform
            # camego2global = np.eye(4, dtype=np.float32)
            # camego2global[:3, :3] = Quaternion(
            #     cam_info['ego2global_rotation']).rotation_matrix
            # camego2global[:3, 3] = cam_info['ego2global_translation']
            # camego2global = torch.from_numpy(camego2global)
            # camego2global_list.append(camego2global) # camego2global is ego2global

            # # camera intrinsics
            # camera_intrinsics = np.eye(4).astype(np.float32)
            # camera_intrinsics[:3, :3] = cam_info["cam_intrinsic"]
            # input_dict["cam_intrinsics"].append(camera_intrinsics)


        # if not self.test_mode:
        #     # annos = self.get_ann_info(index)

        map_geoms = info['annotation'] # annotaions from lidar-center ?
        # map_geoms.keys()
        # dict_keys(['divider', 'ped_crossing', 'boundary', 'centerline'])
      

        map_label2geom = {}
        ann_per =[len(v) for k,v in map_geoms.items()]
      
        for k, v in map_geoms.items():
            if k == 'centerline':
                continue

            if k in ['divider', 'boundary', 'ped_crossing']:
                v = [LineString(p) for p in v]
             
            if k in self.cat2id.keys():
                map_label2geom[self.cat2id[k]] = v

        input_dict = dict(
            location = info['map_location'], # location
            token=info['token'], # 'cfbabc453acc4d5cb6dd759920e1b72a' 

            #pts_filename=info['lidar_path'],
            # lidar_path=info["lidar_path"],
            # sweeps=info['sweeps'],
            img_filenames = image_paths,
            cam_intrinsics = cam_intrinsics,
            cam_extrinsics = cam_extrinsics, 

            ego2img = ego2img_rts, 
            map_geoms = map_label2geom, 

            ego2global_translation=info['ego2global_translation'], # 'e2g_translation': [1272.9167102029883, 2750.0482271163237, 0.0]
            ego2global_rotation=info['ego2global_rotation'], # 'e2g_rotation': [0.9411804770713862, -0.008368539828498084, 0.0007540365189428501, -0.33779980543177557]

            # lidar2ego_translation=info['lidar2ego_translation'],
            # lidar2ego_rotation=info['lidar2ego_rotation'],

         
            scene_name=info['scene_token'], # '055a607b74e04001adef097225a8661a'
    
            sample_idx=info['frame_idx'], # 15
            
        )
        # rotation = Quaternion(input_dict['ego2global_rotation'])
        # translation = input_dict['ego2global_translation']
        # can_bus = input_dict['can_bus']
        # can_bus[:3] = translation
        # can_bus[3:7] = rotation
        # patch_angle = quaternion_yaw(rotation) / np.pi * 180
        # if patch_angle < 0:
        #     patch_angle += 360
        # can_bus[-2] = patch_angle / 180 * np.pi
        # can_bus[-1] = patch_angle

        # lidar2ego = np.eye(4)
        # lidar2ego[:3,:3] = Quaternion(input_dict['lidar2ego_rotation']).rotation_matrix
        # lidar2ego[:3, 3] = input_dict['lidar2ego_translation']
        # ego2global = np.eye(4)
        # ego2global[:3,:3] = Quaternion(input_dict['ego2global_rotation']).rotation_matrix
        # ego2global[:3, 3] = input_dict['ego2global_translation']
        # lidar2global = ego2global @ lidar2ego
        # input_dict['lidar2global'] = lidar2global

        # input_dict['img_filename']
        # ['./data/nuscenes/samples/CAM_FRONT/n015-2018-09-27-15-33-17+0800__CAM_FRONT__1538033900162460.jpg', './data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-09-27-15-33-17+0800__CAM_FRONT_RIGHT__1538033900170339.jpg', './data/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-09-27-15-33-17+0800__CAM_FRONT_LEFT__1538033900154844.jpg', './data/nuscenes/samples/CAM_BACK/n015-2018-09-27-15-33-17+0800__CAM_BACK__1538033900187525.jpg', './data/nuscenes/samples/CAM_BACK_LEFT/n015-2018-09-27-15-33-17+0800__CAM_BACK_LEFT__1538033900197423.jpg', './data/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-09-27-15-33-17+0800__CAM_BACK_RIGHT__1538033900177893.jpg']ta/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-09-27-15-33-17+0800__CAM_BACK_RIGHT__1538033900177893.jpg'

        # len(input_dict['cam_intrinsic']) 6 
        # 

        # input_dict['cam_intrinsic'][0].shape
        # (4, 4)

        # input_dict['ann_info'].keys()
        # dict_keys(['divider', 'ped_crossing', 'boundary', 'centerline'])

        # input_dict.keys()
        # dict_keys(['sample_idx', 'pts_filename', 'lidar_path', 'sweeps', 'ego2global_translation', 'ego2global_rotation', 'lidar2ego_translation', 'lidar2ego_rotation', 'prev_idx', 'next_idx', 'scene_token', 'can_bus', 'frame_idx', 'timestamp', 'map_location', 'lidar2ego', 'camera2ego', 'camera_intrinsics', 'camego2global', 'img_filename', 'lidar2img', 'cam_intrinsic', 'lidar2cam', 'ann_info', 'lidar2global'])
        
        # sample_idx = inf[token]
        
        return input_dict
    

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []

      
        input_dict = self.get_data_info(index)
        # input_dict['ann_info'].keys()
        # dict_keys(['divider', 'ped_crossing', 'boundary', 'centerline'])

        if input_dict is None:
            return None
      
        return self.union2one(data_queue)