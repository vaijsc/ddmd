MODEL_NAME="runwayml/stable-diffusion-v1-5"
data_dir="../datasets/pokemon"
partition="full" # choose between 0 and 1 for dual model training, or full for normal training
output_dir="./experiments/pokemon-rand_caption-partition-$partition"

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" --num_processes=1 \
    --main_process_port 29502 train-ldm.py \
    --pretrained_model_name_or_path $MODEL_NAME \
    --instance_data_dir $data_dir \
    --resolution=512 --gradient_accumulation_steps=1 --center_crop --random_flip \
    --learning_rate=1e-05 --lr_scheduler constant_with_warmup \
    --lr_warmup_steps=1000 \
    --max_train_steps=8001 \
    --train_batch_size=12 \
    --save_steps=200 --modelsavesteps 2000 \
    --output_dir=$output_dir \
    --duplication nodup \
    --class_prompt instancelevel_blip \
    --trainspecial allcaps \
    --instance_prompt_loc ../datasets/pokemon/blip_captions_pokemon_5.json
