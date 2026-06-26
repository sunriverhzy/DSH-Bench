import torch
from PIL import Image
import argparse
import json
import os
import sys
sys.path.append('customization_eval/img_eval/metric')
from metrics_ip.clip_score import CLIPIScore, CLIPTScore
from metrics_ip.dino_score import DINOScore
import glob
from scipy.ndimage import label, find_objects
import os, glob
import json
import statistics
from tqdm import tqdm
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

def process_image(image_path):

    image = Image.open(image_path).convert('RGB')
                
    return image


def find_largest_bounding_box(mask_path):
    # 读取图像
    mask = Image.open(mask_path).convert('L')
    
    # 转换为二值图像
    mask_np = np.array(mask)
    binary_mask = mask_np > 0  # 假设白色部分的像素值大于0
    
    # 标记连通区域
    labeled_mask, num_features = label(binary_mask)
    
    # 找到所有连通区域的边界框
    objects = find_objects(labeled_mask)
    
    # 找到最大的边界框
    max_area = 0
    largest_bbox = None
    for obj in objects:
        y1, x1, y2, x2 = obj[0].start, obj[1].start, obj[0].stop, obj[1].stop
        area = (y2 - y1) * (x2 - x1)
        if area > max_area:
            max_area = area
            largest_bbox = (x1, y1, x2, y2)
    
    return largest_bbox


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--txt_file", type=str, required=False, default='customization_eval/img_eval/taiji_script/result/clip_t/ip_adapter.json', help="txt_file")
    parser.add_argument("--save_file", type=str, required=False, default='test_data_res/ominicontrol_512_base_test_dataset_v0/clip_sim.json', help="txt_file")
    args = parser.parse_args()

    clip_image_metric = CLIPIScore('openai/clip-vit-base-patch32')
    txt_file = args.txt_file
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    score_list = []
    image_pil_list = []
    image_gt_list = []
    for data in tqdm(data_list):
        img1_path =  data["ref_images"][0]["image_path"]
        img2_path_list = data["output_image"]
        for img2_path in img2_path_list:
            try:
                image_1 = process_image(img1_path)
                image_2 = process_image(img2_path)
                image_pil_list.append(image_1)
                image_gt_list.append(image_2)
                score = clip_image_metric.calcul([image_1], [image_2])
                score_list.append([img1_path, img2_path, score])
            except:
                continue
        
    with open(args.save_file, 'w') as f:
        json.dump(score_list, f, indent=2)
    
    