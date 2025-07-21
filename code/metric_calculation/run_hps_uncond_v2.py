from argparse import ArgumentParser
import json
import sys
sys.path.insert(0, "image_quality_test/HPSv2")

import hpsv2
import hpsv2.img_score
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class HPSDataset(Dataset):
    def __init__(self, preprocess_val, image_paths):
        self.preprocess_val = preprocess_val
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_path = self.image_paths[i][0]
        prompt = self.image_paths[i][1]
        pixel_values = self.preprocess_val(Image.open(image_path).convert("RGB"))
        return {
            "pixel_values": pixel_values,
            "image_paths": image_path,
            "prompt": prompt
        }


def main(args):
    device = "cuda"

    hpsv2.img_score.initialize_model()
    model = hpsv2.img_score.model_dict["model"]
    preprocess_val = hpsv2.img_score.model_dict["preprocess_val"]
    checkpoint = torch.load(args.model_name_or_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer = hpsv2.src.open_clip.get_tokenizer("ViT-H-14")
    model = model.to(device)
    model.eval()

    with open(args.txt_file, "r") as f:
        dataset = json.load(f)
    image_paths = []
    for data in tqdm(dataset):
        img_path_list = data["output_image"]
        prompt = data["prompt"]
        for img in img_path_list:
            image_paths.append([img, prompt])
    dataset = HPSDataset(preprocess_val, image_paths)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)


    final_res = []
    for batch in tqdm(dataloader):
        pixel_values = batch["pixel_values"].to(device=device, non_blocking=True)
        image_paths = batch["image_paths"]
        prompt = batch["prompt"]
        input_ids = tokenizer(prompt).to(device=device, non_blocking=True)
        # Use torch.no_grad() for inference
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # Encode text and image features
                text_features = model.encode_text(input_ids, normalize=True)  # (1, emb_size)
                image_features = model.encode_image(pixel_values, normalize=True)

                # Compute logits and scores
                logits_per_image = image_features @ text_features.T
                scores = logits_per_image.squeeze(1).cpu().tolist()

                # Append results
                for image_path, score in zip(image_paths, scores):
                    final_res.append([image_path, score])

    with open(args.save_file, 'w') as f:
        json.dump(final_res, f, indent=2)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--model_name_or_path", type=str, default="ptm/HPSv2/HPS_v2.1_compressed.pt")
    arg_parser.add_argument("--txt_file", type=str, default="image_quality_test/dataset.json")
    arg_parser.add_argument("--save_file", type=str, default="image_quality_test/predictions/hps_uncond_v2.txt")
    arg_parser.add_argument("--batch_size", type=int, default=1)
    arg_parser.add_argument("--num_workers", type=int, default=32)
    args = arg_parser.parse_args()

    main(args)
