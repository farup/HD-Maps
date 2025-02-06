from.base_dataset import BaseMapDataset
from .map_utils.nuscmap_extractor_maptracker import NuscMapExtractor
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
class NuscDatasetMapTracker(BaseMapDataset):
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
    
    def __init__(self, data_root, cam_list=False, **kwargs):
        super().__init__(**kwargs)
        
        self.cam_list = cam_list
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

        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(sample['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = sample['lidar2ego_translation']

        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(sample['e2g_rotation']).rotation_matrix
        ego2global[:3, 3] = sample['e2g_translation']

        lidar2global = ego2global @ lidar2ego
        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_translation = [float(x) for x in lidar2global_translation]
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
  


        # ------------ Swapped In From MapTracker ----------------- 

        map_geoms = self.map_extractor.get_map_geom(location, lidar2global_translation, 
                lidar2global_rotation)

        # map_geoms = self.map_extractor.get_map_geom(location, sample['e2g_translation'], 
        #     sample['e2g_rotation']) # = VectorizedLocalMap.gen_vectorized_samples 
        
        # ---------------------------------------------------------
        lidar_shifted_e2g_translation = np.array(sample['e2g_translation'])
        lidar_shifted_e2g_translation[0] = lidar2global_translation[0]
        lidar_shifted_e2g_translation[1] = lidar2global_translation[1]
        lidar_shifted_e2g_translation = lidar_shifted_e2g_translation.tolist()
        e2g_rotation = sample['e2g_rotation']

        lidar2global = np.eye(4)
        lidar2global[:3,:3] = Quaternion(e2g_rotation).rotation_matrix
        lidar2global[:3, 3] = lidar_shifted_e2g_translation
        global2lidar = np.linalg.inv(lidar2global)
        
        ego2lidar = global2lidar  @ ego2global

     
        map_label2geom = {}
        for k, v in map_geoms.items(): # divider line string, ped cros line string, driv are polygon
          
            # count[k] = [lines, poly]
            if k in self.cat2id.keys():
                map_label2geom[self.cat2id[k]] = v
   
        if self.cam_list: 
                new_cams = {}
                for k,v in sample['cams'].items(): 
                    if k in self.cam_list: 
                        new_cams[k] = v
        else: 
            new_cams = sample['cams']

        ego2img_rts = []
        ego2cam_rts = []
        for c in new_cams.values():
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])

            cam2ego_rt = np.linalg.inv(extrinsic)
            cam2lidar_rt = ego2lidar @ cam2ego_rt
            lidar2cam_rt = np.linalg.inv(cam2lidar_rt)
            ego2cam_rt = lidar2cam_rt

            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

            ego2img_rt = (viewpad @ ego2cam_rt)
            ego2cam_rts.append(ego2cam_rt)
            ego2img_rts.append(ego2img_rt)

            # ego2cam_rt = extrinsic # ego -> cam 
            # viewpad = np.eye(4)
            # viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            # ego2cam_rt = (viewpad @ ego2cam_rt) # ego -> cam_rt  -> img
            # ego2img_rts.append(ego2cam_rt)

        # if sample['sample_idx'] == 0:
        #     is_first_frame = True
        # else:
        #     is_first_frame = self.flag[sample['sample_idx']] > self.flag[sample['sample_idx'] - 1]
        input_dict = {
            'location': location,
            'token': sample['token'],
            'img_filenames': [c['img_fpath'] for c in new_cams.values()],
            # intrinsics are 3x3 Ks
            'cam_intrinsics': [c['intrinsics'] for c in new_cams.values()],
            # extrinsics are 4x4 tranform matrix, **ego2cam**
            'cam_extrinsics': [c['extrinsics'] for c in new_cams.values()],
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