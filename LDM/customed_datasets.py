from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from torchvision.datasets import ImageFolder
import json
import ast
from itertools import chain, repeat, islice
import torch
import numpy as np
import pickle
from pathlib import Path
from datasets import load_from_disk

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def get_classnames(datasetpath):
    if "imagenette_2class" in datasetpath:
        return [ 'church', 'garbage truck']
    else:
        return [ 'tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
        

class ObjectAttributeDataset():
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        partition = None,
        class_prompt=None,
        size=320,
        center_crop=False,
        random_flip = False,
        prompt_json = None,
        duplication = "nodup",
        args = None
    ):
        self.size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.tokenizer = tokenizer
        self.duplication = duplication
        self.trainspecial = args.trainspecial
        self.trainspecial_prob = args.trainspecial_prob
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if self.center_crop else transforms.RandomCrop(size),
                # transforms.RandomHorizontalFlip() if self.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.class_prompt = class_prompt
        if class_prompt in ['instancelevel_blip','instancelevel_ogcap','instancelevel_random']:
            assert prompt_json != None
            
            with open(prompt_json) as f:
                self.prompts = json.load(f)
            self.prompts = list(self.prompts.values())

        self.dataset = load_from_disk(instance_data_root)["train"]
        self.dataset = self.dataset["image"]

        half_size = len(self.dataset) // 2
        if partition == 0:
            self.dataset = self.dataset[:half_size]
            self.prompts = self.prompts[:half_size]
        elif partition == 1:
            self.dataset = self.dataset[half_size:]
            self.prompts = self.prompts[half_size:]

        print("Dataset length: ", len(self.prompts), "Partition: ", partition)
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        instance_image = self.dataset[index]
        example = {}
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.trainspecial is not None:
            if self.trainspecial in ['allcaps']:
                instance_prompt = np.random.choice(self.prompts[index], 1)[0]
            else:
                raise ValueError(f"Invalid trainspecial: {self.trainspecial}")
        else:
            raise NotImplementedError
        example["instance_prompt_ids"] = self.tokenizer(
                instance_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example

def insert_rand_word(sentence,word):
    import random
    sent_list = sentence.split(' ')
    sent_list.insert(random.randint(0, len(sent_list)), word)
    new_sent = ' '.join(sent_list)
    return new_sent