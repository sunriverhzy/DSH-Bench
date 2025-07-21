import torch
from PIL import Image
import argparse
import json
import os
from metrics_ip.clip_score import CLIPIScore, CLIPTScore
from metrics_ip.dino_score import DINOScore
import glob

import os, glob
import json
import statistics
from tqdm import tqdm
from PIL import Image
#import ImageReward as RM

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--txt_file", type=str, required=False, default='customization_eval/img_eval/taiji_script/result/clip_t/ip_adapter.json', help="txt_file")
    parser.add_argument("--save_file", type=str, required=False, default='test_data_res/ominicontrol_512_base_test_dataset_v0/clip_sim.json', help="txt_file")
    args = parser.parse_args()
        
    with open(args.txt_file, 'r') as f:
            data_list = json.load(f)
            
    clip_text_metric_l = CLIPTScore('openai/clip-vit-base-patch32')
    
    final_res = []
    
    score_l_list = []
    score_l = 0
    for data in tqdm(data_list):
        img_path_list = data["output_image"]
        prompt = data["prompt"]
        image_pil_list = []
        prompt_gt_list = []
        for image_path in img_path_list:            
            score = clip_text_metric_l.calcul([Image.open(image_path)], [prompt])
            score_l_list.append([image_path, score])
       
        
    
    with open(args.save_file, 'w') as f:
        json.dump(score_l_list, f, indent=2)
          

    

