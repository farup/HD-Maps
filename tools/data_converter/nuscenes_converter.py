import mmcv
import numpy as np
from os import path as osp
from pyquaternion import Quaternion
import argparse
from nusc_split import TRAIN_SCENES, VAL_SCENES

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

FAIL_SCENES = ['scene-0499', 'scene-0502', 'scene-0515', 'scene-0517']

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        help='specify the root path of dataset')
    parser.add_argument(
        '--newsplit',
        action='store_true')
    parser.add_argument(
        '-v','--version',
        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
        default='v1.0-trainval')
    
    args = parser.parse_args()
    return args

def create_nuscenes_infos_map(root_path,
                            dest_path=None,
                            info_prefix='nuscenes',
                            version='v1.0-trainval',
                            new_split=False):
    """Create info file for map learning task on nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    else:
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    
    if new_split:
        train_scenes = TRAIN_SCENES
        val_scenes = VAL_SCENES

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    
    train_samples, val_samples, test_samples = [], [], []
    
    train_sample_idx = 0
    val_sample_idx = 0
    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP']) # lidar info ( with sample_token)
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token']) # translation and orientation from lidar?
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)
        scene_record = nusc.get('scene', sample['scene_token'])
        log_record = nusc.get('log', scene_record['log_token'])
        location = log_record['location']
        scene_name = scene_record['name']
        if scene_name in FAIL_SCENES: continue
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'cams': {},
            
            'lidar2ego_translation': cs_record['translation'], # list len=3:  [0.943713, 0.0, 1.84023]
            'lidar2ego_rotation': cs_record['rotation'], # list len=3: [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817]
            'e2g_translation': pose_record['translation'], # list len=3: [1010.1328353833223, 610.8111652918716, 0.0]
            'e2g_rotation': pose_record['rotation'], # list len=4: [-0.7495886280607293, -0.0077695335695504636, 0.00829759813869316, -0.6618063711504101]
            'timestamp': sample['timestamp'], # int: 1531883530449377
            'location': location, # 'singapore-onenorth'
            'scene_name': scene_name # 'scene-0001'
        }

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam] # val_saple_idx =1 : cam_front token  'ec7096278e484c9ebe6894a2ad5682e9'
            sd_rec = nusc.get('sample_data', cam_token)
            cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token']) # val_sample_idx=2: cam_front, 'ego_pose_token' ='6ff2a727cdd447c5956582f420c5a80c' ||

            cam2ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix # 3x3
            cam2ego_translation = np.array(cs_record['translation']) # array([1.70079119, 0.01594563, 1.51095764]) # gives the position of the camera origin in the ego frame.  camera's origin and express it in the ego frame, it is located at (1.70, 0.016, 1.51) meters in the ego coordinate system.
            # translation/rotation defined from cam 2 ego 
            ego2cam_rotation = cam2ego_rotation.T
            ego2cam_translation = ego2cam_rotation.dot(-cam2ego_translation) # cam

            transform_matrix = np.eye(4) #ego2cam
            transform_matrix[:3, :3] = ego2cam_rotation
            transform_matrix[:3, 3] = ego2cam_translation

            cam_info = dict(
                extrinsics=transform_matrix, # ego2cam
                intrinsics=cs_record['camera_intrinsic'], # list 3x3
                img_fpath=str(nusc.get_sample_data_path(sd_rec['token'])) # '/cluster/home/terjenf/StreamMapNet/datasets/nuscenes/samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470.jpg' cam_front_right: '/cluster/home/terjenf/StreamMapNet/datasets/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-18-11-07-57+0800__CAM_FRONT_RIGHT__1531883530420339.jpg'
            )
            info['cams'][cam] = cam_info
        
        if scene_name in train_scenes: #
            info.update({
                'sample_idx': train_sample_idx,
                'prev': train_sample_idx - 1,
                'next': train_sample_idx + 1,
            })
            if sample['prev'] == '':
                info['prev'] = -1
            if sample['next'] == '':
                info['next'] = -1
            train_samples.append(info)
            train_sample_idx += 1
        elif scene_name in val_scenes:
            info.update({
                'sample_idx': val_sample_idx,
                'prev': val_sample_idx - 1,
                'next': val_sample_idx + 1,
            })
            if sample['prev'] == '':
                info['prev'] = -1
            if sample['next'] == '':
                info['next'] = -1
            val_sample_idx += 1
            val_samples.append(info)
        else:
            test_samples.append(info)
    
    if dest_path is None:
        dest_path = root_path
    
    if test:
        info_path = osp.join(dest_path, f'{info_prefix}_map_infos_steammapnet_test.pkl')
        print(f'saving test set to {info_path}')
        mmcv.dump(test_samples, info_path)

    else:
        # for training set
        if new_split:
            info_path = osp.join(dest_path, f'{info_prefix}_map_infos_train_steammapnet_newsplit.pkl')
        else:
            info_path = osp.join(dest_path, f'{info_prefix}_map_infos_steammapnet_train.pkl')
        print(f'saving training set to {info_path}')
        mmcv.dump(train_samples, info_path)

        # for val set
        if new_split:
            info_path = osp.join(dest_path, f'{info_prefix}_map_infos_val_steammapnet_newsplit.pkl')
        else:
            info_path = osp.join(dest_path, f'{info_prefix}_map_infos_val_steammapnet.pkl')
        print(f'saving validation set to {info_path}')
        mmcv.dump(val_samples, info_path)


if __name__ == '__main__':
    args = parse_args()

    create_nuscenes_infos_map(root_path=args.data_root, version=args.version, new_split=args.newsplit)