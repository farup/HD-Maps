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


def sort_func(e):
    return e.split("_")[-1]


def sort_func_pred(e):
    return int(e.split("_map_")[-1].split(".")[0])


def get_img_paths(file_path, pred=False): 
    if os.listdir(file_path):

        
        img_paths = [os.path.abspath(os.path.join(file_path, f)) for f in os.listdir(file_path)]

        if pred: 
            img_paths.sort(key=sort_func_pred)
        else:
            img_paths.sort(key=sort_func)

        return img_paths 
       
    
def convert_images_to_video(self, image_files, output_file, fps):
    # Get the list of image files in the input folder

    # Read the first image to get its dimensions
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec for the output video file
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Iterate over each image and write it to the video
    for image_file in image_files:
        
        frame = cv2.imread(image_file)
        video.write(frame)
    
    # Release the video writer and close the video file
    video.release()
    cv2.destroyAllWindows()
    print("Saved video to": output_file)




if __name__ == "__main__": 

    pred_folder = "/cluster/home/terjenf/StreamMapNet/master_work_reversed_fw_coeff_2x0/scene_50/pred"
    cam_folder = ""

    cam_files  = get_img_paths(cam_folder)
    pred_files = get_img_paths(pred_folder, pred=True)

 

    output_file_path = "/cluster/home/terjenf/naplab/data/Trip077/StreamMapNet/video"

    name="master_work_reversed_fw_coeff_2x0_scene_50_pred_video.mp4"

    convert_images_to_video(cam_files, os.path.join(output_file_path, name))


