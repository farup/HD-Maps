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
import pickle


@DATASETS.register_module()
class NapLabDataset(BaseMapDataset):
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
    
    def __init__(self, data_root, fw_coeff_0_start=False,  cam_list=False, **kwargs):
        super().__init__(**kwargs)
        
        self.cam_list = cam_list
        self.data_root = data_root
        self.fw_coeff_0_start = fw_coeff_0_start
        #self.map_extractor = NuscMapExtractor(data_root, self.roi_size)
        self.renderer = Renderer(self.cat2id, self.roi_size, 'naplab')
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.

        """

        start_time = time()
        with open(ann_file, 'rb') as f: 
            ann= pickle.load(f)
            print("Loaded", ann_file)

        samples = list(ann)[::self.interval][self.sample_start:self.sample_end]
        
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
   
             
        # map_geoms = self.map_extractor.get_map_geom(location, sample['e2g_translation'],
                                                    
                                                     
        #         sample['e2g_rotation']) # NuscMapExtractor.get_map_geom
        # map_label2geom = {}
        # for k, v in map_geoms.items(): # divider line string, ped cros line string, driv are polygon
        #     if k in self.cat2id.keys():
        #         map_label2geom[self.cat2id[k]] = v
        
        ego2img_rts = []
        ego2cam_rts = []
        fw_coeffs = []
        cx = []
        cy = []

        # dict_keys(['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'])
        #cam_list = ['CAM_FRONT']

        if self.cam_list: 
            new_cams = {}
            for k,v in sample['cams'].items(): 
                if k in self.cam_list: 
                    new_cams[k] = v
        else: 
            new_cams = sample['cams']

        for c in new_cams.values():
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])
            ego2cam_rt = extrinsic # ego -> cam 
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt) # ego -> cam_rt  -> img
            ego2img_rts.append(ego2cam_rt)

            if self.fw_coeff_0_start: 
                fw_coeffs.append(c['fw_coeff_0_start'])
            else: 
                fw_coeffs.append(c['fw_ceoff'])
            cx.append(c['cx'])
            cy.append(c['cy'])
            ego2cam_rts.append(extrinsic)

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
            'ego2cam': ego2cam_rts,
            'fw_coeff': fw_coeffs, 
            'cx': cx,
            'cy': cy, 
            'map_geoms': None, # {0: List[ped_crossing(LineString)], 1: ...}
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