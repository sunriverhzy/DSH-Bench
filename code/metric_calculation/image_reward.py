# pip install image-reward
import ImageReward as RM
import argparse
import json
import os
from tqdm import tqdm

model = RM.load("customization_eval/img_eval/1st-Place-Solution-in-Google-Universal-Image-Embedding/ImageReward.pt",device='cuda')
#model = RM.load_score(name="CLIP", device="cuda")

def get_reward(prompt, img_path_list):
    rewards = model.score(prompt, img_path_list)
    if isinstance(rewards, float):
        rewards = [rewards]
    return rewards


def get_rank(prompt, img_path):
    ranking, rewards = model.inference_rank(prompt, [img_path])
    print("rank",rewards)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--txt_file", type=str, required=True, default='test_data_res/ominicontrol_512_base_test_dataset_v0/res_detect.json', help="txt_file")
    parser.add_argument("--save_file", type=str, required=True, default='test_data_res/ominicontrol_512_base_test_dataset_v0/clip_sim.json', help="txt_file")

    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    with open(args.txt_file, 'r') as f:
        data_list = json.load(f)
    
    data_res_dict = []
    score = 0
    for data in tqdm(data_list):
        img_path_list = data["output_image"]
        prompt = data["prompt"]
        raw_score_list = get_reward(prompt, img_path_list)
        raw_score = sum(raw_score_list)/len(raw_score_list)
        data_res_dict.append([img_path_list,raw_score])
        score += raw_score
    
    print(score/len(data_res_dict))
        
    with open(args.save_file, 'w') as f:
        json.dump(data_res_dict, f, indent=2)

