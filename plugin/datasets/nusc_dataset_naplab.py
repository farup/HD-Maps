from.base_dataset import BaseMapDataset
from .map_utils.nuscmap_extractor import NuscMapExtractor
from mmdet.datasets import DATASETS
import numpy as np
from .visualize.renderer import Renderer
import mmcv
from time import time
from pyquaternion import Quaternion
import pickle

#from NAPLab_car.tools.data_processing.naplab import NapLab


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
    
    def __init__(self, cam_list=False, **kwargs):
        super().__init__(**kwargs)
        self.cam_list = cam_list
      
  

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

        #ann = mmcv.load(ann_file)
        samples = list(ann)[::self.interval][self.sample_start:self.sample_end]
        
        print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
        self.samples = samples
    
    def load_matching(self, matching_file):
        with open(matching_file, 'rb') as pf:
            data = pickle.load(pf)
        total_samples = 0
        for scene_name, info in data.items():
            total_samples += len(info['sample_ids'])
        assert total_samples == len(self.samples), 'Matching info not matched with data samples'
        self.matching_meta = data
        print(f'loaded matching meta for {len(data)} scenes')
    

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

            fw_coeffs.append(c['fw_coeff'])
            cx.append(c['cx'])
            cy.append(c['cy'])
            ego2cam_rts.append(extrinsic)


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
            #'ego2global_translation': sample['e2g_translation'], 
            #'ego2global_rotation': Quaternion(sample['e2g_rotation']).rotation_matrix.tolist(),
            'ego2global_translation': sample['e2g_translation'], 
            'ego2global_rotation': Quaternion(sample['e2g_rotation']).rotation_matrix.tolist(),
            'sample_idx': sample['sample_idx'],
            'scene_name': sample['scene_name'],
            'lidar2ego_translation': sample['C1_front60Single2ego_translation'],
            'lidar2ego_rotation': sample['C1_front60Single2ego_rotation'],
        }

        return input_dict

    def get_sample_old(self, idx):
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
        lidar2ego[:3,:3] = Quaternion(sample['C1_front60Single2ego_rotation']).rotation_matrix 
        lidar2ego[:3, 3] = sample['C1_front60Single2ego_translation'] 

        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(sample['e2g_rotation']).rotation_matrix
        ego2global[:3, 3] = sample['e2g_translation']

        # NOTE: The original StreamMapNet uses the ego location to query the map,
        # to align with the lidar-centered setting in MapTR, we made some modifiactions 
        # here to switch to the lidar-center setting
        lidar2global = ego2global @ lidar2ego
        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_translation = [float(x) for x in lidar2global_translation]
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

      
        lidar_shifted_e2g_translation = np.array(sample['e2g_translation'])
        lidar_shifted_e2g_translation[0] = lidar2global_translation[0]
        lidar_shifted_e2g_translation[1] = lidar2global_translation[1]
        lidar_shifted_e2g_translation = lidar_shifted_e2g_translation.tolist() # ego to global but x,y of lidar?
        e2g_rotation = sample['e2g_rotation']

        lidar2global = np.eye(4)
        lidar2global[:3,:3] = Quaternion(e2g_rotation).rotation_matrix
        lidar2global[:3, 3] = lidar_shifted_e2g_translation
        global2lidar = np.linalg.inv(lidar2global)
        
        ego2lidar = global2lidar  @ ego2global # ego2lidar but only in 

    
        if self.cam_list: 
            new_cams = {}
            for k,v in sample['cams'].items(): 
                if k in self.cam_list: 
                    new_cams[k] = v
        else: 
            new_cams = sample['cams']
        
        ego2img_rts = []
        ego2cam_rts = []
        fw_coeffs = []
        cx = []
        cy = []
        for c in new_cams.values():
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])

            # ego coord to cam coord
            #ego2cam_rt = extrinsic
            fw_coeffs.append(c['fw_coeff'])
            cx.append(c['cx'])
            cy.append(c['cy'])

            cam2ego_rt = np.linalg.inv(extrinsic)
            cam2lidar_rt = ego2lidar @ cam2ego_rt
            lidar2cam_rt = np.linalg.inv(cam2lidar_rt)
            ego2cam_rt = lidar2cam_rt

            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

            ego2img_rt = (viewpad @ ego2cam_rt)
            ego2cam_rts.append(ego2cam_rt)
            ego2img_rts.append(ego2img_rt)


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
            #'ego2global_translation': sample['e2g_translation'], 
            #'ego2global_rotation': Quaternion(sample['e2g_rotation']).rotation_matrix.tolist(),
            'ego2global_translation': lidar_shifted_e2g_translation, 
            'ego2global_rotation': Quaternion(e2g_rotation).rotation_matrix.tolist(),
            'sample_idx': sample['sample_idx'],
            'scene_name': sample['scene_name'],
            'lidar2ego_translation': sample['C1_front60Single2ego_translation'],
            'lidar2ego_rotation': sample['C1_front60Single2ego_rotation'],
        }

        return input_dict




if __name__ == "__main__": 

    # meta info for submission pkl
    ann_file='/cluster/home/terjenf/NAPLab_car/data/Trip077/naplab_infos.pkl',
    meta = dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False,
        output_format='vector')

    coords_dim = 2

    cam_list = False
    roi_size = (60, 30)
    cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
    }
    # data processing pipelines
    test_pipeline = [
        dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
        dict(type='ResizeMultiViewImages',
            size=img_size, # H, W
            change_intrinsics=True,
            ),
        dict(type='Normalize3D', **img_norm_cfg),
        dict(type='PadMultiViewImages', size_divisor=32),
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['img'], meta_keys=(
            'token', 'ego2img', 'sample_idx', 'ego2global_translation',
            'ego2global_rotation', 'img_shape', 'scene_name'))
    ]

    eval_config = dict(
    type='NuscDataset',
    data_root='/cluster/home/terjenf/maptracker/datasets/nuscenes',
    ann_file=ann_file,
    meta=meta,
    roi_size=roi_size,
    cat2id=cat2id,
    pipeline=[
        dict(
            type='VectorizeMap',
            coords_dim=coords_dim,
            simplify=True,
            normalize=False,
            roi_size=roi_size
        ),
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['vectors',], meta_keys=['token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'])
    ],
    interval=1,
    )


    test=dict( 
        ann_file=ann_file,
        meta=meta,
        cam_list=cam_list, 
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
    )


    naplab_dataset = NapLabDataset(cam_list=False, **test)
 