CUDA_VISIBLE_DEVICES=0 python DistillMD.py --train --logdir ./experiments/CIFAR10-distill --dataset_root ../datasets \
--dataset CIFAR10 --img_size 32 --batch_size 128 --fid_cache ./stats/cifar10.train.npz --total_steps 780001 \
--teacher_ckpt1 ./experiments/CIFAR10-dual/ckpt-step780000_model_0.pt \
--teacher_ckpt2 ./experiments/CIFAR10-dual/ckpt-step780000_model_1.pt
