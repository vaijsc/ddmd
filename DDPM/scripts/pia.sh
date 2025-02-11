CUDA_VISIBLE_DEVICES=0 python PIA/attack.py \
    --checkpoint ./experiments/CIFAR10-distill/ckpt-step780000.pt \
    --dataset cifar10 \
    --attacker_name PIA \
    --attack_num 10 \
    --interval 20

CUDA_VISIBLE_DEVICES=0 python PIA/attack.py \
    --checkpoint ./experiments/CIFAR10-distill/ckpt-step780000.pt \
    --dataset cifar10 \
    --attacker_name PIAN \
    --attack_num 10 \
    --interval 20