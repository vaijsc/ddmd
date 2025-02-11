CUDA_VISIBLE_DEVICES=0 python SecMIA/secmia.py --model_dir ./experiments/CIFAR10-distill --dataset_root ../datasets \
--dataset cifar10 --t_sec 100 --k 10 --model_name ckpt-step780000.pt
