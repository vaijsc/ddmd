from transformers import DeiTFeatureExtractor, DeiTModel, DeformableDetrModel, AutoImageProcessor, BeitModel, EfficientFormerModel, ViTModel
import torch
from datasets import Dataset, load_from_disk
from torchvision import transforms
from PIL import Image
import os
import argparse
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="get image embedding")
    parser.add_argument("--data_dir",type=str,default=None)
    parser.add_argument("--sample_file",type=str,default=None)
    parser.add_argument("--membership",type=int,default=None)
    parser.add_argument("--img_num",type=int,default=None)
    parser.add_argument("--gpu",type=int,default=None)
    parser.add_argument("--save_dir",type=str,default=None)
    parser.add_argument("--method",type=str,default="cosine")
    parser.add_argument("--image_encoder",type=str,default="deit")
    parser.add_argument("--similarity_score_dim",type=int,default=1)
    parser.add_argument("--fraction",type=str,default="member")
    args = parser.parse_args()

    return args

def compute_scores(emb_one, emb_two,method):
    """Computes cosine similarity between two vectors."""
    if method == "cosine":
        scores = torch.nn.functional.cosine_similarity(emb_one, emb_two, dim=args.similarity_score_dim)
    elif method == "euclidean":
        scores = torch.nn.functional.pairwise_distance(emb_one, emb_two, dim=args.similarity_score_dim)
    elif method == "manhattan":
        scores = torch.sum(torch.abs(emb_one - emb_two), dim=args.similarity_score_dim)
    elif method == "hamming":
        emb_one = emb_one.int()
        emb_two = emb_two.int()
        scores = torch.sum((emb_one ^ emb_two), dim=args.similarity_score_dim)
    return scores.cpu().detach().numpy()

def get_scores(args, model, dataset, sample_file, save_dir, feature_extractor, device):
    scores=[]
    for i in tqdm(range(len(dataset["image"]))):
        image_target = dataset["image"][i].convert("RGB")
        inputs = feature_extractor(image_target, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states_target = outputs.last_hidden_state
        temp = []
        for j in range(args.img_num):
            filename = f"image_{i+1:02}_{j+1:02}.jpg"
            save_path = os.path.join(sample_file, filename)
            img = Image.open(save_path).convert("RGB")
            inputs = feature_extractor(img, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            temp.append(compute_scores(last_hidden_states_target,last_hidden_states,args.method)[0])
        temp = np.array(temp)
        average = np.mean(temp,axis = 0)
        temp = [average.tolist()]
        temp.append(args.membership)
        scores.append(temp)
    torch.save(scores, save_dir)

def load_pokemon_datasets(dataset_root):
    dataset = load_from_disk(os.path.join(dataset_root, 'pokemon'))
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    return train_dataset, test_dataset

def main():

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # dataset = Dataset.from_dict(torch.load(args.data_dir))
    train_dataset, test_dataset = load_pokemon_datasets(args.data_dir)

    if args.image_encoder=="deit":
        feature_extractor  = DeiTFeatureExtractor.from_pretrained("facebook/deit-base-distilled-patch16-384")
        model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-384", add_pooling_layer=False)
    elif args.image_encoder=="detr":
        feature_extractor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
        model = DeformableDetrModel.from_pretrained("SenseTime/deformable-detr") 
    elif args.image_encoder=="beit":
        feature_extractor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k") 
    elif args.image_encoder=="eformer":
        feature_extractor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
        model = EfficientFormerModel.from_pretrained("snap-research/efficientformer-l1-300")  
    elif args.image_encoder=="vit":
        feature_extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    else:
        raise ValueError("Image encoder must be one of deit, detr, beit, eformer, vit")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    if args.membership == 1:
        sample_file = os.path.join(args.sample_file, "members")
        save_dir = os.path.join(args.save_dir, "members.pt")
        get_scores(args, model, train_dataset, sample_file, save_dir, feature_extractor, device)
    elif args.membership == 0:
        sample_file = os.path.join(args.sample_file, "non-members")
        save_dir = os.path.join(args.save_dir, "non-members.pt")
        get_scores(args, model, test_dataset, sample_file, save_dir, feature_extractor, device)
    else:
        raise ValueError("Membership must be either 1 or 0")

if __name__ == "__main__":
    args = parse_args()
    main()


