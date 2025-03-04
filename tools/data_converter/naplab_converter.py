import numpy as np
from os import path as osp
from pyquaternion import Quaternion
import argparse

import pickle

import sys 

import os
print(os.getcwd())


sys.path.append("/cluster/home/terjenf/")
sys.path.append("/cluster/home/terjenf/NAPLab_car")
sys.path.append("/cluster/home/terjenf/NAPLab_car/tools")
sys.path.append("/cluster/home/terjenf/NAPLab_car/tools/naplab")


from NAPLab_car.tools.naplab_api.naplab import NapLab

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        help='specify the root path of dataset')
    parser.add_argument(
        '--trip',
        default='Trip077')
    
    args = parser.parse_args()
    return args

def create_naplab_infos_map(root_path,
                            dest_path=None,
                            info_prefix='naplab',
                            trip='Trip077'):
    """Create info file for map learning task on nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
    """

    
    naplab = NapLab(dataroot=root_path, trip=trip, verbose=True)

    train_sample_idx = 0
    train_samples = []


    for i, sample in enumerate(naplab.sample):

        lidar_token = None
        sd_rec = naplab.get('sample_data', sample['data']['C1_front60Single'])
        cs_record = naplab.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = naplab.get('ego_pose', sd_rec['ego_pose_token'])


        lidar_path = "" 

        #mmcv.check_file_exist(lidar_path)

        #scene_record = naplab.get('scene', sample['scene_token'])
        log_record = None
        location = "Trondheim"
        scene_name = trip
     
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'cams': {},
            
            'C1_front60Single2ego_translation': cs_record['translation'],
            'C1_front60Single2ego_rotation': cs_record['rotation'],
            'e2g_translation': pose_record['translation'],
            'e2g_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'location': location,
            'scene_name': scene_name
        }

        # obtain 6 image's information per frame
        # camera_types = [
        #     'CAM_FRONT',
        #     'CAM_FRONT_RIGHT',
        #     'CAM_FRONT_LEFT',
        #     'CAM_BACK',
        #     'CAM_BACK_LEFT',
        #     'CAM_BACK_RIGHT',
        # ]

        camera_types = [
        'C1_front60Single',
        'C8_R2',
        'C7_L2',
        'C4_rearCam', 
        'C6_L1',
        'C5_R1',]
        
        for cam in camera_types:
            cam_token = sample['data'][cam]
            sd_rec = naplab.get('sample_data', cam_token)
            cs_record = naplab.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])

            cam2ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
            cam2ego_translation = np.array(cs_record['translation'])

            ego2cam_rotation = cam2ego_rotation.T
            ego2cam_translation = ego2cam_rotation.dot(-cam2ego_translation)

            transform_matrix = np.eye(4) #ego2cam
            transform_matrix[:3, :3] = ego2cam_rotation
            transform_matrix[:3, 3] = ego2cam_translation

            cam_info = dict(
                extrinsics=transform_matrix, # ego2cam
                intrinsics=cs_record['camera_intrinsics'],
                img_fpath=sd_rec['filename']
            )
            info['cams'][cam] = cam_info
        
        
        info.update({
            'sample_idx': train_sample_idx,
            'prev': train_sample_idx - 1,
            'next': train_sample_idx + 1,
        })

        if i == 0:
            info['prev'] = -1
        
        if i == (len(naplab.sample) -1):
            info['next'] = -1
        train_samples.append(info)
        train_sample_idx += 1
    
    if dest_path is None:
        dest_path = root_path
    
    
    info_path = osp.join(dest_path, trip, f'{info_prefix}_infos.pkl')
    print(f'saving naplab set to {info_path}')
    with open(info_path, 'wb') as f: 
        pickle.dump(train_samples, f)
        print("Saved to", info_path) 

if __name__ == '__main__':
    args = parse_args()

    create_naplab_infos_map(root_path=args.data_root, trip=args.trip)