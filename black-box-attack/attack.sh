CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--unet_path="../LDM/experiments/pokemon-rand_caption-partitionfull_instancelevel_blip_nodup_special_allcaps_0.1/checkpoint_6000" \
--num_validation_images=3 \
--inference=50 \
--data_dir="../datasets" \
--save_dir="./gen-pokemon-aug-6k" \
--seed=1337 \
--fraction="member" \
--ddim

CUDA_VISIBLE_DEVICES=0 python3 cal_embedding.py \
--data_dir="../datasets" \
--sample_file="./gen-pokemon-aug-6k" \
--membership=1 \
--img_num=3 \
--gpu=0 \
--save_dir="distance-pokemon-aug-6k" \

CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--unet_path="../LDM/experiments/pokemon-rand_caption-partitionfull_instancelevel_blip_nodup_special_allcaps_0.1/checkpoint_6000" \
--num_validation_images=3 \
--inference=50 \
--data_dir="../datasets" \
--save_dir="./gen-pokemon-aug-6k" \
--seed=1337 \
--fraction="non-member" \
--ddim

CUDA_VISIBLE_DEVICES=0 python3 cal_embedding.py \
--data_dir="../datasets" \
--sample_file="./gen-pokemon-aug-6k" \
--membership=0 \
--img_num=3 \
--gpu=0 \
--save_dir="distance-pokemon-aug-6k" \

CUDA_VISIBLE_DEVICES=0 python3 attack.py \
--target_member_dir="distance-pokemon-distill-30/members.pt" \
--target_non_member_dir="distance-pokemon-distill-30/non-members.pt" \
--method="distribution"


# CUDA_VISIBLE_DEVICES=3 accelerate launch inference-corrected.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --unet1_path="/vinai/baotq4/projects/mia-defense/experiments/ldm-pokemon/alt-teachers-30k/model1" \
# --unet2_path="/vinai/baotq4/projects/mia-defense/experiments/ldm-pokemon/alt-teachers-30k/model2" \
# --num_validation_images=3 \
# --inference=50 \
# --data_dir="/vinai/baotq4/projects/SecMI-LDM/datasets" \
# --save_dir="./gen-pokemon-corrected-50-ddpm" \
# --seed=1337 \
# --fraction="member" \
# --ddim