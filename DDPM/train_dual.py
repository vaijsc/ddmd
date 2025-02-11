import copy
import json
import os
import sys

import numpy as np
import warnings
import torch
from tensorboardX import SummaryWriter
import torchvision
from torchvision.utils import make_grid, save_image
from torchvision.datasets import CIFAR10, CIFAR100, CelebA, SVHN
from tqdm import trange
from torch.utils.data import Subset
from mia_evals.dataset_utils import MIACelebA, MIACIFAR10, MIACIFAR100, MIASVHN, MIASTL10, MIAImageFolder
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score

import torch.multiprocessing as mp
import torch.distributed as dist

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Dual Training for Diffusion Models')
    
    # Add arguments (previously flags)
    parser.add_argument('--train', action='store_true', help='train from scratch')
    parser.add_argument('--eval', action='store_true', help='load ckpt.pt and evaluate FID and IS')
    parser.add_argument('--ch', type=int, default=128, help='base channel of UNet')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 2, 2, 2], help='channel multiplier')
    parser.add_argument('--attn', nargs='+', type=int, default=[1], help='add attention to these levels')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='# resblock in each level')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate of resblock')
    parser.add_argument('--beta_1', type=float, default=1e-4, help='start beta value')
    parser.add_argument('--beta_T', type=float, default=0.02, help='end beta value')
    parser.add_argument('--T', type=int, default=1000, help='total diffusion steps')
    parser.add_argument('--mean_type', type=str, default='epsilon', choices=['xprev', 'xstart', 'epsilon'], help='predict variable')
    parser.add_argument('--var_type', type=str, default='fixedlarge', choices=['fixedlarge', 'fixedsmall'], help='variance type')
    parser.add_argument('--lr', type=float, default=2e-4, help='target learning rate')
    parser.add_argument('--grad_clip', type=float, default=1., help="gradient norm clipping")
    parser.add_argument('--total_steps', type=int, default=800001, help='total training steps')
    parser.add_argument('--img_size', type=int, default=32, help='image size')
    parser.add_argument('--warmup', type=int, default=5000, help='learning rate warmup')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='workers of Dataloader')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help="ema decay rate")
    parser.add_argument('--parallel', action='store_true', help='multi gpu training')
    parser.add_argument('--resume_from_ckpt1', type=str, default=None, help='path to the checkpoint to resume from')
    parser.add_argument('--resume_from_ckpt2', type=str, default=None, help='path to the checkpoint to resume from')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='data set')
    parser.add_argument('--dataset_root', type=str, default='./datasets', help='data set')
    parser.add_argument('--logdir', type=str, default='./logs/DDPM_CIFAR10_EPS', help='log directory')
    parser.add_argument('--sample_size', type=int, default=64, help="sampling size of images")
    parser.add_argument('--sample_step', type=int, default=1000, help='frequency of sampling')
    parser.add_argument('--save_step', type=int, default=30000, help='frequency of saving checkpoints, 0 to disable during training')
    parser.add_argument('--eval_step', type=int, default=0, help='frequency of evaluating model, 0 to disable during training')
    parser.add_argument('--num_images', type=int, default=25000, help='the number of generated images for evaluation')
    parser.add_argument('--fid_use_torch', action='store_true', help='calculate IS and FID on gpu')
    parser.add_argument('--fid_cache', type=str, default='./stats/cifar10.train.npz', help='FID cache')
    parser.add_argument('--defense', type=str, default=None, help='Defense tricks, e.g., DP-SGD or L2 Regularizationgg')
    parser.add_argument('--only_member', action='store_true', help='Training only on member split')
    
    # New arguments for dual training
    parser.add_argument('--num_gpus', type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--split_ratio', type=float, default=0.5, help='ratio to split the dataset')
    
    return parser.parse_args()

device = torch.device('cuda:0')

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))

def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x

def warmup_lr(step):
    # return min(step, args.warmup) / args.warmup
    return min(step, 5000) / 5000

def evaluate(sampler, model, save_path, ckpt_name):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, args.num_images, args.batch_size, desc=desc):
            batch_size = min(args.batch_size, args.num_images - i)
            x_T = torch.randn((batch_size, 3, args.img_size, args.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    # save sampled images
    torch.save({'samples': images}, os.path.join(save_path, f'{ckpt_name}_samples.pt'))
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, args.fid_cache, num_images=args.num_images,
        use_torch=args.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images

def get_dataset(args, only_member=False):
    dataset_root = args.dataset_root
    if args.dataset.upper() != 'CIFAR10-SYNTHETIC':
        splits = np.load(f'./mia_evals/member_splits/{args.dataset.upper()}_train_ratio0.5.npz')
        member_idxs = splits['mia_train_idxs']
    else:
        member_idxs = None

    if args.dataset.upper() == 'CIFAR10':
        transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                      (0.5, 0.5, 0.5))])
        if only_member:
            dataset = MIACIFAR10(member_idxs, root=os.path.join(dataset_root, 'cifar10'), train=True,
                                 transform=transforms, download=True)
        else:
            dataset = CIFAR10(root=os.path.join(dataset_root, 'cifar10'), train=True,
                              transform=transforms)
    elif args.dataset.upper() == 'CIFAR100':
        transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                      (0.5, 0.5, 0.5))])
        if only_member:
            dataset = MIACIFAR100(member_idxs, root=os.path.join(dataset_root, 'cifar100'), train=True,
                                  transform=transforms)
        else:
            dataset = CIFAR100(root=os.path.join(dataset_root, 'cifar100'), train=True,
                               transform=transforms)
    elif args.dataset.upper() == 'CELEBA':
        # for CelebA, first center crop 140 and then resize to 32 (by default)
        transforms = torchvision.transforms.Compose([
            # torchvision.transforms.CenterCrop(140),
            torchvision.transforms.Resize(args.img_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])
        if only_member:
            dataset = MIACelebA(member_idxs, root=os.path.join(dataset_root, 'celeba'), split='train',
                                transform=transforms, download=True)
        else:
            dataset = CelebA(root=os.path.join(dataset_root, 'celeba'), split='train',
                             transform=transforms, download=True)
    elif args.dataset.upper() == 'SVHN':
        transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                      (0.5, 0.5, 0.5))])
        if only_member:
            dataset = MIASVHN(member_idxs, root=os.path.join(dataset_root, 'svhn'), split='train',
                              transform=transforms, download=True)
        else:
            dataset = SVHN(root=os.path.join(dataset_root, 'svhn'), split='train',
                           transform=transforms, download=True)
    elif args.dataset.upper() == 'STL10_U':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])
        if only_member:
            dataset = MIASTL10(member_idxs, root=os.path.join(dataset_root, 'stl10'), split='unlabeled',
                               download=True, transform=transforms)
        else:
            dataset = torchvision.datasets.STL10(root=os.path.join(dataset_root, 'stl10'), split='unlabeled',
                                                 download=True, transform=transforms)
    elif args.dataset.upper() == 'TINY-IN':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])
        if only_member:
            dataset = MIAImageFolder(member_idxs, root=os.path.join(dataset_root, 'tiny-imagenet-200/train'),
                                     transform=transforms)
        else:
            dataset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_root, 'tiny-imagenet-200/train'),
                                                       transform=transforms)
    else:
        raise NotImplemented

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    return dataset, data_loader

def get_split_datasets(args, only_member=False):
    full_dataset, _ = get_dataset(args, only_member=only_member)
    dataset_size = len(full_dataset)
    split_index = int(dataset_size * args.split_ratio)
    
    generator = torch.Generator().manual_seed(1234)
    indices = torch.randperm(dataset_size, generator=generator)
    dataset1 = Subset(full_dataset, indices[:split_index])
    dataset2 = Subset(full_dataset, indices[split_index:])
    
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, shuffle=False, drop_last=True)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    return dataloader1, dataloader2

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_model(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Create datasets and dataloaders
    dataloader1, dataloader2 = get_split_datasets(args, only_member=args.only_member)
    datalooper = infiniteloop(dataloader1 if rank % 2 == 0 else dataloader2)

    # Create model, optimizer, scheduler, trainer, and sampler for this GPU
    model = UNet(T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
                 num_res_blocks=args.num_res_blocks, dropout=args.dropout).to(device)

    if args.resume_from_ckpt1:
        if rank == 0:
            print(f'Resuming from checkpoint: {args.resume_from_ckpt1}')
            ckpt = torch.load(args.resume_from_ckpt1, map_location=device)
        else:
            ckpt = torch.load(args.resume_from_ckpt2, map_location=device)
        model.load_state_dict(ckpt['net_model'])
        skip_step = ckpt['step']
        args.total_steps = args.total_steps - skip_step
        if rank == 0:
            print(f"Resuming from step: {skip_step}, Remaining training steps: {args.total_steps}")

    if args.defense == 'L2':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(model, args.beta_1, args.beta_T, args.T).to(device)
    sampler = GaussianDiffusionSampler(model, args.beta_1, args.beta_T, args.T, args.img_size,
                                       args.mean_type, args.var_type).to(device)

    ema_model = copy.deepcopy(model)
    ema_sampler = GaussianDiffusionSampler(ema_model, args.beta_1, args.beta_T, args.T, args.img_size,
                                           args.mean_type, args.var_type).to(device)

    # log setup
    if not os.path.exists(os.path.join(args.logdir, 'sample')):
        os.makedirs(os.path.join(args.logdir, 'sample'))
    x_T = torch.randn(args.sample_size, 3, args.img_size, args.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader1))[0][:args.sample_size]) + 1) / 2
    writer = SummaryWriter(args.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(args.logdir, "flagfile.txt"), 'w') as f:
        f.write(str(args))
    # show model size
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # Training loop
    pbar = trange(args.total_steps, dynamic_ncols=True) if rank == 0 else range(args.total_steps)
    for step in pbar:
        optim.zero_grad()
        x_0 = next(datalooper).to(device)
        loss = trainer(x_0).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        sched.step()
        ema(model, ema_model, args.ema_decay)

        if rank == 0:
            writer.add_scalar(f'loss_model_{rank}', loss.item(), step)
            pbar.set_postfix(loss='%.3f' % loss.item())

        # Sample
        if args.sample_step > 0 and step % args.sample_step == 0:
            model.eval()
            with torch.no_grad():
                x_0 = ema_sampler(x_T)
                grid = (make_grid(x_0) + 1) / 2
                if args.resume_from_ckpt1:
                    path = os.path.join(args.logdir, 'sample', f'model_{rank}_{step + skip_step}.png')
                else:
                    path = os.path.join(args.logdir, 'sample', f'model_{rank}_{step}.png')
                save_image(grid, path)
                if rank == 0:
                    writer.add_image(f'sample_model_{rank}', grid, step)
            model.train()

        # Save
        if args.save_step > 0 and step > 0 and step % args.save_step == 0:
            ckpt = {
                'net_model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'sched': sched.state_dict(),
                'optim': optim.state_dict(),
                'step': step + skip_step if args.resume_from_ckpt1 else step,
                'x_T': x_T,
            }
            if args.resume_from_ckpt1:
                torch.save(ckpt, os.path.join(args.logdir, f'ckpt-step{step + skip_step}_model_{rank}.pt'))
            else:
                torch.save(ckpt, os.path.join(args.logdir, f'ckpt-step{step}_model_{rank}.pt'))

        # Evaluate
        if args.eval_step > 0 and step > 0 and step % args.eval_step == 0:
            if rank == 0:
                net_IS, net_FID, _ = evaluate(sampler, model, args.logdir, f'model_{rank}_{step}')
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model, args.logdir, f'model_{rank}_{step}_ema')
                metrics = {
                    f'IS_model_{rank}': net_IS[0],
                    f'IS_std_model_{rank}': net_IS[1],
                    f'FID_model_{rank}': net_FID,
                    f'IS_EMA_model_{rank}': ema_IS[0],
                    f'IS_std_EMA_model_{rank}': ema_IS[1],
                    f'FID_EMA_model_{rank}': ema_FID
                }
                pbar.write(
                    f"Model {rank} - {step}/{args.total_steps} " +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(args.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()  
    cleanup()

def train():
    world_size = args.num_gpus
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=train_model, args=(rank, world_size, args))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

def eval():
    for i in range(args.num_gpus):
        device = torch.device(f'cuda:{i}')
        
        model = UNet(T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
                     num_res_blocks=args.num_res_blocks, dropout=args.dropout).to(device)
        sampler = GaussianDiffusionSampler(model, args.beta_1, args.beta_T, args.T, img_size=args.img_size,
                                           mean_type=args.mean_type, var_type=args.var_type).to(device)
        
        ckpt_name = f'ckpt-step800000_model_{i}.pt'
        ckpt = torch.load(os.path.join(args.logdir, ckpt_name))
        model.load_state_dict(ckpt['net_model'])
        (IS, IS_std), FID, samples = evaluate(sampler, model, args.logdir, f'model_{i}_{ckpt_name.split(".")[0]}')
        print(f"Model {i}     : IS:{IS:6.3f}({IS_std:.3f}), FID:{FID:7.3f}")
        save_image(torch.tensor(samples[:256]),
                   os.path.join(args.logdir, f'samples_model_{i}.png'),
                   nrow=16)
        
        model.load_state_dict(ckpt['ema_model'])
        (IS, IS_std), FID, samples = evaluate(sampler, model, args.logdir, f'model_{i}_{ckpt_name.split(".")[0]}_ema')
        print(f"Model {i}(EMA): IS:{IS:6.3f}({IS_std:.3f}), FID:{FID:7.3f}")
        save_image(torch.tensor(samples[:256]),
                   os.path.join(args.logdir, f'{ckpt_name}samples_ema_model_{i}.png'),
                   nrow=16)

def main():
    global args
    args = parse_arguments()
    
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if args.train:
        train()
    if args.eval:
        eval()
    if not args.train and not args.eval:
        print('Add --train and/or --eval to execute corresponding tasks')

if __name__ == '__main__':
    main()
