import json
import os
from tqdm import tqdm

"""
[
    {
        "ref_image": {
            "image_path": ""
            "height": "",
            "width": "",
        },
        "prompt": "",
        "subject": "",
        "metadata": {
            "subject_difficulty_level": "",
            "prompt_scenario": "",
            "cate1": "",
            "cate2": "",
            "cate3": ""
        }
    }
]
"""

cate2en = {
    "realistic": "Photorealistic",
    "nonrealistic": "Non-Photorealistic",
    "物体":"Object",
    "动物":"Animal",
    "人":"Person",
    "交通工具":"Vehicle",
    "乐器":"Musical Instrument",
    "公共设施":"Public Facility",
    "食品饮料":"Food and Beverage",
    "医疗用品":"Medical Supply",
    "图书":"Book",
    "家具":"Furniture",
    "家电":"Home Appliance",
    "两栖动物":"Amphibian",
    "建筑物":"Building",
    "数码产品":"Digital Product",
    "昆虫":"Insect",
    "文具":"Stationery",
    "日用品":"Daily Necessity",
    "植物":"Plant",
    "珠宝首饰":"Jewelry",
    "美妆护肤":"Beauty and Skincare",
    "艺术品":"Artwork",
    "衣服":"Clothing",
    "运动器具":"Sports Equipment",
    "鞋包配饰":"Shoe, Bag and Accessory",
    "玩具":"Toy",
    "哺乳动物":"Mammal",
    "爬行类":"Reptile",
    "鸟":"Bird",
    "鱼":"Fish",
    "半身或全身":"Half-body or Full-body Photo",
    "面部特写":"Facial Close-up",
    "艺术形象":"Artistic Image and Celebrity",
    "艺术形象或名人":"Artistic Image and Celebrity",
}




path = "/apdcephfs_nj7/share_3004949/mapleshu/NIPS_2025/data/final_data_0522.json"

data = json.load(open(path))
res = {}
for i in tqdm(range(len(data))):
    item = data[i]
    image_path = item["ref_images"][0]["image_path"]
    basename = cate2en[item["metadata"]["cate1"]] + "_" + cate2en[item["metadata"]["cate2"]] + "_" + "_".join(os.path.basename(image_path).replace(" ", "-").split("_")[2:]).replace("-", "_")
    basename = os.path.basename(image_path)
    output_path = os.path.join("./images", basename)
    
    if output_path not in res:
        os.system("cp \"{}\" \"{}\"".format(image_path, output_path))
        res[output_path] = {
            "ref_image": {
                "image_path": output_path,
                "height": item["ref_images"][0]["height"],
                "width": item["ref_images"][0]["width"],
            },
            "prompt": [
                {
                    "prompt": item["prompt"],
                    "prompt_scenario": item["metadata"]["prompt_level"]
                }
            ],
            "subject": item["entities"],
            "metadata": {
                "subject_difficulty_level": item["metadata"]["subject_hard_level"],
                "cate1": cate2en[item["metadata"]["cate1"]],
                "cate2": cate2en[item["metadata"]["cate2"]],
                "cate3": cate2en[item["metadata"]["cate3"]],
            }
        }
    else:
        res[output_path]["prompt"].append(
            {
                "prompt": item["prompt"],
                "prompt_scenario": item["metadata"]["prompt_level"]
            }
        )




with open("./DSH-Bench_image_info.json", 'w') as wt:
    json.dump(res, wt, indent=2, ensure_ascii=False)


with open("./DSH-Bench_image_info.json", 'r') as wt:
    data = json.load(wt)
    cnt = 0
    for key in data:
        if len(data[key]["prompt"]) > 12:
            print("wrong")
            cnt += 1
    print(cnt)
