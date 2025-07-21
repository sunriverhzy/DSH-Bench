import torch
from PIL import Image
import argparse
import json
import os
import sys
sys.path.append('customization_eval/img_eval/metric')
from metrics_ip.clip_score import CLIPIScore, CLIPTScore
from metrics_ip.dinov2_score import DINOV2Score
import glob

import os, glob
import json
import statistics
from tqdm import tqdm
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None

def apply_mask_and_paste_on_white_background(image, mask):
        
    mask_array = np.array(mask)
    white_background = Image.new('RGB', image.size, (0, 0, 0))
    image_array = np.array(image)
    alpha_channel = np.where(mask_array > 128, 255, 0).astype(np.uint8)
    rgba_image_array = np.dstack((image_array, alpha_channel))
    processed_image = Image.fromarray(rgba_image_array, 'RGBA')
    white_background.paste(processed_image, (0, 0), processed_image)
    return white_background
    
def process_image_with_bbox(image_path, mask_path, bbox, size):
    if bbox != [] and bbox is not None:
        cx, cy, w, h = bbox

        x1 = cx - w / 2
        y1 = cy - h / 2
        
        x2 = cx + w / 2
        y2 = cy + h / 2
        h, w = size

        
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)
    
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        #image = apply_mask_and_paste_on_white_background(image, mask)  
            
        cropped_image = image.crop((x1, y1, x2, y2)).resize((224,224))
    else:
        image = Image.open(image_path).convert('RGB')
                
        cropped_image = image

    return cropped_image

def process_image(image_path):

    image = Image.open(image_path).convert('RGB')
                
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--txt_file", type=str, required=False, default='customization_eval/img_eval/taiji_script/result/clip_t/ip_adapter.json', help="txt_file")
    parser.add_argument("--save_file", type=str, required=False, default='test_data_res/ominicontrol_512_base_test_dataset_v0/clip_sim.json', help="txt_file")
    args = parser.parse_args()


    clip_image_metric = DINOV2Score('dinov2_vitl14')
    txt_file = args.txt_file
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    score = 0
    image_pil_list = []
    image_gt_list = []
    score_list = []

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
    