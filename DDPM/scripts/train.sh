CUDA_VISIBLE_DEVICES=0 python train.py --train --logdir ./experiments/CIFAR10 --dataset_root ../datasets \
--dataset CIFAR10 --img_size 32 --batch_size 128 --fid_cache ./stats/cifar10.train.npz --total_steps 780001 \
--resume_from_ckpt ./experiments/CIFAR10/ckpt-step450000.pt \