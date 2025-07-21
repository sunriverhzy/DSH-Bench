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


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--txt_file", type=str, required=False, default='customization_eval/img_eval/taiji_script/result/clip_t/ip_adapter.json', help="txt_file")
    parser.add_argument("--save_file", type=str, required=False, default='test_data_res/ominicontrol_512_base_test_dataset_v0/clip_sim.json', help="txt_file")
    args = parser.parse_args()


    clip_image_metric = CLIPIScore('openai/clip-vit-large-patch14-336')
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