# pip install image-reward
import os, glob
import json
import statistics

import ImageReward as RM
model = RM.load("ImageReward-v1.0")
# rewards = model.score("<prompt>", ["<img1_obj_or_path>", "<img2_obj_or_path>", ...])

def get_image_paths(folder):
    return [f for f in glob.glob(os.path.join(folder, "*")) if f.endswith((".png", ".jpg", ".jpeg"))]

keys, prompts = [], []
with open("prompts/evaluations/jsons/eval_benchmark30.json", "r") as f:
    data = json.load(f)
for data_i in data:
    key = data_i['index'].split("_")[0]
    keys.append(key)
    prompts.append(data_i["text"])

# image_root_path = "metrics/baseline_results/ip_adapter_result_sdxl"
# image_root_path = "metrics/baseline_results/ip_adapter_result_sd15"
# image_root_path = "metrics/baseline_results/custom_diffusion_result"
# image_root_path = "metrics/baseline_results/moma_result"
# image_root_path = "metrics/baseline_results/dreammatcher/custom_diffusion_sd15"
image_root_path = "metrics/baseline_results/dreammatcher/dreammatcher_sd15"
scores = []

for key, prompt in zip(keys, prompts):
    print(key, prompt)
    image_paths = os.path.join(image_root_path, key)
    image_paths_list = get_image_paths(image_paths)

    reward_score =  model.score(prompt, image_paths_list)
    reward_score_mean = statistics.mean(reward_score)

    scores.append(reward_score_mean)

print(image_root_path, ", ", " image reward score: ", statistics.mean(scores))