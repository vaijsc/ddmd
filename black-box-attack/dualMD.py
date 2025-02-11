import argparse
import os
from tqdm import tqdm
import torch
import torch.utils.checkpoint
from datasets import Dataset, load_from_disk
from torchvision import transforms
from diffusers import UNet2DConditionModel, DDIMScheduler
from defense_pipeline import StableDiffusionPipeline
from PIL import Image

from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images) : return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    ) 
    parser.add_argument(
        "--unet1_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet2_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=3
    )
    parser.add_argument("--inference", type=int, default=100)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337
    )
    parser.add_argument(
        "--fraction",
        type=str,
        default="member"
    )
    parser.add_argument(
        "--ddim",
        action="store_true"
    )
    args = parser.parse_args()

    return args

def generate(args, dataset, save_dir, pipeline, generator):
    for i in tqdm(range(len(dataset["text"]))):
        for j in range(args.num_validation_images):
            image = pipeline(dataset["text"][i], num_inference_steps=args.inference, guidance_scale=7.5, generator=generator).images[0]
            filename = f"image_{i+1:02}_{j+1:02}.jpg"
            save_path = os.path.join(save_dir, filename)
            image.save(save_path)

def main():
    args = parse_args()

    mem_dir = os.path.join(args.save_dir, "members")
    non_mem_dir = os.path.join(args.save_dir, "non-members")
    if not os.path.exists(mem_dir):
        os.makedirs(mem_dir)
    if not os.path.exists(non_mem_dir):
        os.makedirs(non_mem_dir)

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, revision=None, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False
    )
    # Disable the progress bar for the pipeline
    pipeline.set_progress_bar_config(disable=True)
    if args.ddim:
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.unet = UNet2DConditionModel.from_pretrained(args.unet1_path, subfolder="unet", torch_dtype=torch.float16)
    pipeline.unet_alt = UNet2DConditionModel.from_pretrained(args.unet2_path, subfolder="unet", torch_dtype=torch.float16)
    pipeline.to("cuda")
    pipeline.unet_alt.to("cuda")

    generator = torch.Generator("cuda").manual_seed(args.seed)

    def load_pokemon_datasets(dataset_root):
        dataset = load_from_disk(os.path.join(dataset_root, 'pokemon'))
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        return train_dataset, test_dataset

    train_dataset, test_dataset = load_pokemon_datasets(args.data_dir)

    if args.fraction == "member":
        generate(args, train_dataset, mem_dir, pipeline, generator)
    else:
        generate(args, test_dataset, non_mem_dir, pipeline, generator)

if __name__ == "__main__":
    main()