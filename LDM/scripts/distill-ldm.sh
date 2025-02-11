MODEL_NAME="runwayml/stable-diffusion-v1-5"
data_dir="../datasets/pokemon"
output_dir="./experiments/pokemon-rand_caption-distill-6k"

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" --num_processes=1 \
    --main_process_port 29505 distill-ldm.py \
    --unet_path1 "./experiments/pokemon-rand_caption-partition-0_instancelevel_blip_nodup_special_allcaps_0.1/checkpoint_6000" \
    --unet_path2 "./experiments/pokemon-rand_caption-partition-1_instancelevel_blip_nodup_special_allcaps_0.1/checkpoint_6000" \
    --pretrained_model_name_or_path $MODEL_NAME \
    --instance_data_dir $data_dir \
    --resolution=512 --gradient_accumulation_steps=1 --center_crop --random_flip \
    --learning_rate=1e-05 --lr_scheduler constant_with_warmup \
    --lr_warmup_steps=1000 \
    --max_train_steps=20001 \
    --train_batch_size=12 \
    --save_steps=200 --modelsavesteps 2000 \
    --output_dir=$output_dir \
    --duplication nodup \
    --class_prompt instancelevel_blip \
    --trainspecial allcaps \
    --instance_prompt_loc ../datasets/pokemon/blip_captions_pokemon_5.json \