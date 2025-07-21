import json

import numpy as np
from scipy import stats
import glob
from scipy.stats import spearmanr, pearsonr, kendalltau
import pandas as pd
import krippendorff

def calculate_correlations(x, y):
    """
    Calculate both Pearson correlations with detailed steps

    Parameters:
    x, y: arrays of equal length containing paired observations

    Returns:
    dict: Detailed results of correlation calculations
    """

    def pearson_manual(x, y):
        "formula: r = Σ((x - μx)(y - μy)) / √(Σ(x - μx)² * Σ(y - μy)²)"

        # Step 1: Calculate means
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Step 2: Calculate deviations
        x_dev = x - x_mean
        y_dev = y - y_mean

        # Step 3: Calculate sum of products of deviations
        sum_prod_dev = np.sum(x_dev * y_dev)

        # Step 4: Calculate sum of squared deviations
        sum_sq_dev_x = np.sum(x_dev**2)
        sum_sq_dev_y = np.sum(y_dev**2)

        # Step 5: Calculate correlation coefficient
        r = sum_prod_dev / np.sqrt(sum_sq_dev_x * sum_sq_dev_y)
        return r

    manual_res = pearson_manual(x, y)
    scipy_res = stats.pearsonr(x, y)[0]
    assert np.isclose(manual_res, scipy_res), "Manual calculation and SciPy verification do not match"

    return round(manual_res, 4)


def get_gpt4_score(img_list):
    gpt4_list = glob.glob('processed/LLaMA-Factory/process_subject/gpt4_test/human_v1/output_*.jsonl')
    dict_id = {}
    img_dict={}
    for gpt4_path in gpt4_list:
        with open(gpt4_path, 'r') as f:
            output_lines = f.readlines()
            f.close()
        gpt4_intput_path = gpt4_path.replace('output','input')
        with open(gpt4_intput_path, 'r') as f:
            input_lines = f.readlines()
            for output_line in output_lines:
                output_line = json.loads(output_line)
                dict_id[output_line["custom_id"]] = int(output_line["content"].split('Score: ')[-1])
            
            for input_line in input_lines:
                input_line = json.loads(input_line)
                tgt_img_path = input_line["tgt_img_path"]
                request_id = input_line["custom_id"]
                img_dict[tgt_img_path] = dict_id[request_id]


    fin_res = []
    for i in img_list:
        fin_res.append(img_dict[i])

    rating = np.array(list(fin_res))

    return rating


def get_human_rating(file, image_list):
    with open(file, 'r') as f:
        data_list = json.load(f)

    fin_data = []
    for data in data_list:
        img = data["images"][1]
        if img not in image_list:
            continue
        score = json.loads(data["conversations"][1]["value"])["score"]
        fin_data.append(float(score))


    print(len(fin_data), fin_data[:10])
    rating = np.array(list(fin_data))

    return rating


    
def get_rating(file, image_list):
    with open(file, 'r') as f:
        data_list = json.load(f)

    fin_data = []
    index_list_id = []
    index = -1
    img_dict = {}
    for data in data_list:
        score = data[-1]
        img = data[1]
        index += 1
        if img not in image_list:
            continue
        img_dict[img] = float(score)
        index_list_id.append(index)

    for image in image_list:
        fin_data.append(img_dict[image])

    rating = np.array(list(fin_data))

    # 1. 获取img_dict的key列表
    dict_keys = list(img_dict.keys())

    # 2. 找到img_list中每个元素在dict_keys中的index
    index_list_id = [dict_keys.index(img) for img in image_list]

    return rating, index_list_id

def get_ir_rating(file, image_list):
    with open(file, 'r') as f:
        data_list = json.load(f)

    fin_data = []
    img_dict = {}
    for data in data_list:
        score = data[-1]
        img = data[0]

        img_dict[img] = float(score)

    for image in image_list:
        fin_data.append(img_dict[image])
        
    rating = np.array(list(fin_data))

    return rating


def get_llm_rating(file, method):

    input_json_path = 'dataset/subject_consistent/after_biaozhu/all_method_sample_test_qwen_1600.json'
    with open(input_json_path, 'r') as f:
        input_data_list = json.load(f)

    fin_data = []
    image_list = []
    index_list = []
    index = -1
    for key, data in input_data_list.items():
        
        images = data[1]
        method_img = images.split('/')[-3]+'_'+images.split('/')[-2]
        if method not in method_img:
            continue
        fin_data.append(float(data[2]))
        image_list.append(data[1])




    rating = np.array(list(fin_data))

    return rating,image_list,index_list


def get_llm_rating_qwen_with_correct(file, method):
    with open(file, 'r') as f:
        data_list = f.readlines()

    input_json_path = 'dataset/subject_consistent/after_biaozhu/all_method_sample_test.json'
    with open(input_json_path, 'r') as f:
        input_data_list = json.load(f)

    fin_data = []
    image_list = []
    index_list = []
    index = -1

    input_json_path = 'dataset/subject_consistent/after_biaozhu/all_method_sample_test_qwen_1600.json'
    #input_json_path = 'processed/LLaMA-Factory/saves/qwen2_vl-7b-subject/lora/sft_0509_dataset_v6/test/predict-0509-v5_sample_4800/lost_part_qwen_dict.json'
    with open(input_json_path, 'r') as f:
        image_dict = json.load(f)


    for input_data_list,data in zip(input_data_list,data_list):
        index += 1
        images = input_data_list["images"][1]
        method_img = images.split('/')[-3]+'_'+images.split('/')[-2]
        if method not in images:
            continue
        try:
            data = json.loads(data)
            predict = data["predict"]
            score = int(json.loads(predict)["score"])
            #if score==1:
            #    continue
            #if 'index_select_samples/split' not in input_data_list["images"][1]:
            #    continue
            
            
            fin_data.append(float(score))
            image_list.append(images)
            index_list.append(index)

        except:
            #continue
            try:
                fin_data.append(image_dict[images][-1])
                image_list.append(images)
                index_list.append(index)
            except:
                continue
    print(len(fin_data))
    rating = np.array(list(fin_data))

    return rating,image_list,index_list

def get_llm_rating_qwen(file, method):
    with open(file, 'r') as f:
        data_list = f.readlines()

    input_json_path = 'dataset/subject_consistent/after_biaozhu/all_method_sample_test.json'
    with open(input_json_path, 'r') as f:
        input_data_list = json.load(f)

    fin_data = []
    image_list = []
    index_list = []
    index = -1
    for input_data_list,data in zip(input_data_list,data_list):
        index += 1
        try:
            data = json.loads(data)
            predict = data["predict"]
            score = int(json.loads(predict)["score"])
            #if score==1:
            #    continue
            #if 'index_select_samples/split' not in input_data_list["images"][1]:
            #    continue
            
            images = input_data_list["images"][1]
            method_img = images.split('/')[-3]+'_'+images.split('/')[-2]
            if method not in method_img:
                continue
            fin_data.append(float(score))
            image_list.append(images)
            index_list.append(index)

        except:
            continue

    rating = np.array(list(fin_data))

    return rating,image_list,index_list


def get_krippendorff_rating(human_rating, metric_rating):
    data = np.array([metric_rating, human_rating])

    alpha = krippendorff.alpha(reliability_data=data, level_of_measurement='interval')
    return alpha


def get_kendalltau(human_rating, metric_rating):
    return kendalltau(human_rating, metric_rating)[0]

def get_spearmanr(human_rating, metric_rating):
    return spearmanr(human_rating, metric_rating)[0]

def get_method(method):
    result={}
    llm_rating,image_list,index_list = get_llm_rating_qwen_with_correct('processed/LLaMA-Factory/saves/qwen2_vl-7b-subject/lora/sft_0509_dataset_v6/test/predict-0509-v5_sample_4800/generated_predictions.jsonl', method)
    human_rating,_ = get_rating('processed/img_eval/dreambench_plus/ft_local/ours_human_rating_correct.json',image_list)

    dino_rating,_ = get_rating('customization_eval/img_eval/result/ours_human_testdata_v1.json/dino_sim_seg.json',image_list)
    dino_v2_rating,_ = get_rating('customization_eval/img_eval/result/ours_human_testdata_v1.json/dino_v2.json',image_list)
    dreamsim_rating,_ = get_rating('customization_eval/img_eval/result/ours_human_testdata_v1.json/dreamsim.json',image_list)
    image_retrieval_rating = get_ir_rating('customization_eval/img_eval/result/ours_human_testdata_v1.json/image_retrieval_score.json',image_list)
    clip_i_b_rating,_ = get_rating('customization_eval/img_eval/result/ours_human_testdata_v1.json/clip_sim_b.json',image_list)
    clip_i_l_rating,index_list_id = get_rating('customization_eval/img_eval/result/ours_human_testdata_v1.json/clip_sim_l.json',image_list)
    gpt4_rating = get_gpt4_score(image_list)

    results = calculate_correlations(human_rating, dino_rating)
    print("dino",results)
    result["dino_person"] = results
    results = calculate_correlations(human_rating, dino_v2_rating)
    print("dinov2",results)
    result["dinov2_person"] = results
    
    results = calculate_correlations(human_rating, dreamsim_rating)
    print("dreamsim_rating",results)
    result["dreamsim_person"] = results
    results = calculate_correlations(human_rating, image_retrieval_rating)
    result["image_retrieval_person"] = results
    print("image_retrieval_rating",results)

    results = calculate_correlations(human_rating, clip_i_b_rating)
    print("clip_i_b_rating",results)
    result["clip_i_b_person"] = results
    results = calculate_correlations(human_rating, clip_i_l_rating)
    result["clip_i_l_person"] = results
    print("clip_i_l_rating",results)


    results = calculate_correlations(human_rating, gpt4_rating)
    result["gpt4_person"] = results
    print("gpt4_rating",results)
    results = calculate_correlations(human_rating, llm_rating)
    print("llm",results)
    result["llm_person"] = results

    human_rating = human_rating/5
    gpt4_rating = gpt4_rating/4
    llm_rating = llm_rating/5

    """ result["dino_krippendorff"] = get_krippendorff_rating(dino_rating, human_rating)
    result["dinov2_krippendorff"] = get_krippendorff_rating(dino_v2_rating, human_rating)
    result["dreamsim_krippendorff"] = get_krippendorff_rating(dreamsim_rating, human_rating)
    result["image_retrieval_krippendorff"] = get_krippendorff_rating(image_retrieval_rating, human_rating)
    result["clip_i_b_krippendorff"] = get_krippendorff_rating(clip_i_b_rating, human_rating)
    result["clip_i_l_krippendorff"] = get_krippendorff_rating(clip_i_l_rating, human_rating)
    result["gpt4_krippendorff"] = get_krippendorff_rating(gpt4_rating, human_rating)
    result["llm_krippendorff"] = get_krippendorff_rating(llm_rating, human_rating) """
    

    #gpt4_rating = gpt4_rating * (5/4)


    
    result["dino_kendalltau"] = get_kendalltau(dino_rating, human_rating)
    result["dinov2_kendalltau"] = get_kendalltau(dino_v2_rating, human_rating)
    result["dreamsim_kendalltau"] = get_kendalltau(dreamsim_rating, human_rating)
    result["image_retrieval_kendalltau"] = get_kendalltau(image_retrieval_rating, human_rating)
    result["clip_i_b_kendalltau"] = get_kendalltau(clip_i_b_rating, human_rating)
    result["clip_i_l_kendalltau"] = get_kendalltau(clip_i_l_rating, human_rating)
    result["gpt4_kendalltau"] = get_kendalltau(gpt4_rating, human_rating)
    result["llm_kendalltau"] = get_kendalltau(llm_rating, human_rating)


    result["dino_spearmanr"] = get_spearmanr(dino_rating, human_rating)
    result["dinov2_spearmanr"] = get_spearmanr(dino_v2_rating, human_rating)
    result["dreamsim_spearmanr"] = get_spearmanr(dreamsim_rating, human_rating)
    result["image_retrieval_spearmanr"] = get_spearmanr(image_retrieval_rating, human_rating)
    result["clip_i_b_spearmanr"] = get_spearmanr(clip_i_b_rating, human_rating)
    result["clip_i_l_spearmanr"] = get_spearmanr(clip_i_l_rating, human_rating)
    result["gpt4_spearmanr"] = get_spearmanr(gpt4_rating, human_rating)
    result["llm_spearmanr"] = get_spearmanr(llm_rating, human_rating)


    print(result)

    
    return result
    

method_list = ['uno','blip','dreambooth','textual','custom_diffusion','ominicontrol','real','adapter','emu2','lambda','ms_diffusion','ssr_encoder','ominigen','ViCo','NeTI_origin','lambda-eclipse', '/HiPer/','']
final_res = {}
for method in method_list:
    print('********************************')
    print(method)
    result = get_method(method)
    final_res[method] = result

df = pd.DataFrame.from_dict(final_res, orient='index')

# 保存为csv
df.to_csv('result_krippendorff_interval.csv')