import argparse
import mmcv
from mmcv import Config
import os
from mmdet3d.datasets import build_dataset, build_dataloader
from IPython import embed
import imageio

import sys 

sys.path.append("/cluster/home/terjenf/naplab")
# sys.path.append("/cluster/home/terjenf/naplab/naplab")
# sys.path.append("/cluster/home/terjenf/naplab/naplab/naplab")


from naplab import NapLab


sys.path.append("/cluster/home/terjenf/")
sys.path.append("/cluster/home/terjenf/StreamMapNet/")
sys.path.append("/cluster/home/terjenf/StreamMapNet/plugin")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('idx', type=int,
        help='which scene to visualize')
    parser.add_argument('--result', 
        default=None,
        help='prediction result to visualize'
        'If submission file is not provided, only gt will be visualized')
    parser.add_argument('--thr', 
        type=float,
        default=0.4,
        help='score threshold to filter predictions')
    parser.add_argument(
        '--out-dir', 
        default='demo',
        help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args

def import_plugin(cfg):
    '''
        import modules from plguin/xx, registry will be update
    '''

    import sys
    sys.path.append(os.path.abspath('.'))    
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            
            def import_path(plugin_dir):
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            plugin_dirs = cfg.plugin_dir
            if not isinstance(plugin_dirs, list):
                plugin_dirs = [plugin_dirs,]
            for plugin_dir in plugin_dirs:
                import_path(plugin_dir)

    
def sort_func(e):
    return e.split("_")[-1]


def sort_func_pred(e):
    return int(e.split("_map_")[-1].split(".")[0])



def main():

    dataroot ="/cluster/home/terjenf/naplab/data"
    trip="Trip077"
    nap = NapLab(dataroot=dataroot, trip=trip)
    args = parse_args()


    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    # build the dataset
    dataset = build_dataset(cfg.eval_config)

    # ann_file = mmcv.load('datasets/nuScenes/nuscenes_map_infos_val.pkl')
    scene_name2idx = {}
    for idx, sample in enumerate(dataset.samples):
        scene = sample['scene_name']
        if scene not in scene_name2idx:
            scene_name2idx[scene] = []
        scene_name2idx[scene].append(idx)

    scene_name = sorted(list(scene_name2idx.keys()))[args.idx]
    print(scene_name)
    scene_dir = os.path.join(args.out_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    start_idx = scene_name2idx[scene_name][0]
    results = mmcv.load(args.result)



    pred_dir = os.path.join(scene_dir, 'pred')
    file_name_cam = os.path.join(nap.dataroot, nap.trip, "StreamMapNet", "plots")
    if os.listdir(file_name_cam):

        pred_dir = "/cluster/home/terjenf/StreamMapNet/master_work_reversed_fw_coeff_2x0/scene_50/pred"

        cam_imgs = [os.path.abspath(os.path.join(file_name_cam, f)) for f in os.listdir(file_name_cam)]
        pred_dirs = [os.path.abspath(os.path.join(pred_dir, f)) for f in os.listdir(pred_dir)]

        cam_imgs.sort(key=sort_func)
        pred_dirs.sort(key=sort_func_pred)

        #nap.convert_images_to_video(image_files=cam_imgs, output_file="/cluster/home/terjenf/naplab/data/Trip077/StreamMapNet/video/cam_video_2.mp4", fps=10)
        nap.convert_images_to_video(image_files=pred_dirs, output_file="/cluster/home/terjenf/naplab/data/Trip077/StreamMapNet/video/pred_video_6.mp4", fps=10)
    
 
    cam_imgs = []
    pred_imgs = []

    for idx, sample in zip(scene_name2idx[scene_name], dataset.samples):
        
        sample_id = nap._token2ind.get('sample')[sample['token']]

        viz_image = nap.visualize_sample_compact(sample_id, model_name="StreamMapNet", save=True)
        cam_imgs.append(viz_image)

        #out_dir = os.path.join(scene_dir, str(idx - start_idx + 1))
        #gt_dir = os.path.join(scene_dir, 'gt')
        

        if args.result is not None:
            os.makedirs(pred_dir, exist_ok=True)
            pred_img = dataset.show_result_video(
                    submission=results, 
                    idx=idx, 
                    score_thr=args.thr, 
                    out_dir=pred_dir,
                    viz_image=viz_image,
                )



        pred_imgs.append(pred_img)
            
        #os.makedirs(gt_dir, exist_ok=True)
        #dataset.show_gt(idx, gt_dir)
    
   

 


if __name__ == '__main__':
    main()