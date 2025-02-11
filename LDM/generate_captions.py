## works with aug2 env
# python caption_images.py --path /fs/cml-projects/diffusion_rep/data/imagenette2-320/train/ --num_caps 20
# python caption_images.py --path /fs/cml-projects/diffusion_rep/data/laion_10k_random/train/ --num_caps 20
# python caption_images.py --path /fs/cml-projects/diffusion_rep/data/laion_10k_random_aesthetics_5plus/train/ --num_caps 20

from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from lavis.models import load_model_and_preprocess
from datasets import load_from_disk
import os
import glob
from itertools import groupby
import argparse
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def load_demo_image(img_path, image_size, device):
    raw_image = Image.open(img_path).convert('RGB')   
    w,h = raw_image.size
    # display(raw_image.resize((w//5,h//5)))
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

def load_pokemon_datasets(dataset_root):
    dataset = load_from_disk(dataset_root)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    return train_dataset

def capgen(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = args.im_size
    np.random.seed(args.seed)
    # model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    # model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    # model.eval()
    # model = model.to(device)

    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

    dataset = load_pokemon_datasets(args.path)

    caption_data = defaultdict(list)

    for idx, image in enumerate(tqdm(dataset["image"])):
        caption_data[idx].append(dataset["text"][idx])
        image = image.convert("RGB")
        image = vis_processors["eval"](image).unsqueeze(0).to(device)
        for i in range(args.num_caps):
            with torch.no_grad():
                # beam search
                # caption = model.generate(image, sample=False, num_beams=3, max_length=50, min_length=5) 
                # nucleus sampling
                caption = model.generate({"image": image}, use_nucleus_sampling=True, top_p=0.9, max_length=20, min_length=5) 
                # print('caption: '+caption[0])
                
            caption_data[idx].append(caption[0])
        
        # if (idx+1) % 50 == 0:
        #     # break
        #     print(f"{idx+1}/{len(dataset)} done!")
        #     print(f"latest caption: {caption[0]}")

    # clean_captions = {}
    # # clean up repeated words
    # for image, caption in caption_data.items():
    #     clean_captions[image] = ' '.join(item[0] for item in groupby(caption.split()))
    
    with open(os.path.join(args.path, f'blip_captions_pokemon_{args.num_caps}.json'), 'w') as f:
        json.dump(caption_data, f, indent=4, sort_keys=False)


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', 
                        default='/fs/cml-projects/diffusion_rep/data/imagenette_2class/train/',
                        type=str, help='path to the image folder')
    parser.add_argument('--num_caps', type=int, default=1)
    parser.add_argument('--im_size', type=int, default=384)
    parser.add_argument('--seed', type=int, default=11111)
    
    args = parser.parse_args()
    capgen(args)