import copy
import json
import os
import sys

import numpy as np
import warnings
from absl import app, flags
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torchvision.datasets import CIFAR10, CIFAR100, CelebA, SVHN
from torch.utils.data import Subset
from tensorboardX import SummaryWriter
import torchvision
from SecMIA.dataset_utils import MIACelebA, MIACIFAR10, MIACIFAR100, MIASVHN, MIASTL10, MIAImageFolder
from tqdm import trange
from diffusion import extract, GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800001, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', True, help='multi gpu training')

flags.DEFINE_string('dataset', 'CIFAR10', help='data set')
flags.DEFINE_string('dataset_root', './datasets', help='dataset')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_string('teacher_ckpt1', './experiments/CIFAR10-dual/ckpt-step800000_model_0.pt', help='teacher checkpoint')
flags.DEFINE_string('teacher_ckpt2', './experiments/CIFAR10-dual/ckpt-step800000_model_1.pt', help='teacher checkpoint')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 30000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 25000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_string('defense', None, help='Defense tricks, e.g., DP-SGD or L2 Regularizationgg')
flags.DEFINE_bool('only_member', True, help='Training only on member split')

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
    return min(step, FLAGS.warmup) / FLAGS.warmup


def evaluate(sampler, model, save_path, ckpt_name):
    model.eval()
    with torch.no_grad():
        images = []
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc="generating images"):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    # save sampled images
    torch.save({'samples': images}, os.path.join(save_path, f'{ckpt_name}_samples.pt'))
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(images, FLAGS.fid_cache, num_images=FLAGS.num_images,
                                                    use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def get_dataset(FLAGS, only_member=False):
    dataset_root = FLAGS.dataset_root
    if FLAGS.dataset.upper() != 'CIFAR10-SYNTHETIC':
        splits = np.load(f'./mia_evals/member_splits/{FLAGS.dataset.upper()}_train_ratio0.5.npz')
        member_idxs = splits['mia_train_idxs']
    else:
        member_idxs = None

    if FLAGS.dataset.upper() == 'CIFAR10':
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
    elif FLAGS.dataset.upper() == 'CIFAR100':
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
    elif FLAGS.dataset.upper() == 'CELEBA':
        # for CelebA, first center crop 140 and then resize to 32 (by default)
        transforms = torchvision.transforms.Compose([
            # torchvision.transforms.CenterCrop(140),
            torchvision.transforms.Resize(FLAGS.img_size),
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
    elif FLAGS.dataset.upper() == 'SVHN':
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
    elif FLAGS.dataset.upper() == 'STL10_U':
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
    elif FLAGS.dataset.upper() == 'TINY-IN':
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

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True,
                                              num_workers=FLAGS.num_workers)

    return dataset, data_loader

class GaussianDiffusionTrainer(torch.nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, teacher_model):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        with torch.no_grad():
            target = teacher_model(x_t, t)
        loss = F.mse_loss(self.model(x_t, t), target, reduction='none')
        return loss
    
def get_split_datasets(FLAGS, dataset, only_member=False):
    dataset_size = len(dataset)
    split_index = int(dataset_size * 0.5)
    
    generator = torch.Generator().manual_seed(1234)
    indices = torch.randperm(dataset_size, generator=generator)
    dataset1 = Subset(dataset, indices[:split_index])
    dataset2 = Subset(dataset, indices[split_index:])
    
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=FLAGS.batch_size, shuffle=False, drop_last=True)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=FLAGS.batch_size, shuffle=False, drop_last=True)
    
    return dataloader1, dataloader2

def train():
    dataset, dataloader = get_dataset(FLAGS, only_member=FLAGS.only_member)
    subloader1, subloader2 = get_split_datasets(FLAGS, dataset, only_member=FLAGS.only_member)
    
    # Student Model
    student_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    
    # Teacher Model (Pre-trained)
    teacher_model1 = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
    teacher_model2 = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
    
    # Load pre-trained teacher weights
    teacher_ckpt1 = torch.load(FLAGS.teacher_ckpt1, map_location=device)
    teacher_model1.load_state_dict(teacher_ckpt1['net_model'])
    teacher_model1.eval()
    teacher_model1.requires_grad = False
    teacher_ckpt2 = torch.load(FLAGS.teacher_ckpt2, map_location=device)
    teacher_model2.load_state_dict(teacher_ckpt2['net_model'])
    teacher_model2.eval()
    teacher_model2.requires_grad = False

    # EMA Model for the student
    ema_student = copy.deepcopy(student_model)

    # Optimizer and LR Scheduler for Student
    if FLAGS.defense == 'L2':
        print('Applying L2 Regularization')
        optim = torch.optim.Adam(student_model.parameters(), lr=FLAGS.lr, weight_decay=5e-4)
    else:
        optim = torch.optim.Adam(student_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # Samplers for Student and Teacher
    student_sampler = GaussianDiffusionSampler(student_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
                                               FLAGS.mean_type, FLAGS.var_type).to(device)

    # EMA Sampler
    ema_sampler = GaussianDiffusionSampler(ema_student, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
                                           FLAGS.mean_type, FLAGS.var_type).to(device)
    
    trainer = GaussianDiffusionTrainer(student_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    
    # Data loader loop
    # datalooper = infiniteloop(dataloader)
    datalooper1 = infiniteloop(subloader1)
    datalooper2 = infiniteloop(subloader2)

    # Setup logging
    if not os.path.exists(os.path.join(FLAGS.logdir, 'sample')):
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()

    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size).to(device)
    
    # Backup flags
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in student_model.parameters():
        model_size += param.data.nelement()
    print('Student model params: %.2f M' % (model_size / 1024 / 1024))
    model_size = 0
    for param in teacher_model1.parameters():
        model_size += param.data.nelement()
    print('Teacher model params: %.2f M' % (model_size / 1024 / 1024))

    # Start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            if step % 2 == 0:
                x_0 = next(datalooper1).to(device)
                loss = trainer(x_0, teacher_model2).mean()
            else:
                x_0 = next(datalooper2).to(device)
                loss = trainer(x_0, teacher_model1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()

            ema(student_model, ema_student, FLAGS.ema_decay)

            # Log training loss
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # Sampling and saving
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                student_model.eval()
                with torch.no_grad():
                    sample_images = ema_sampler(x_T)
                    grid = (make_grid(sample_images) + 1) / 2
                    save_image(grid, os.path.join(FLAGS.logdir, 'sample', f'{step}.png'))
                    writer.add_image('sample', grid, step)
                student_model.train()

            # Save checkpoint
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': student_model.state_dict(),
                    'ema_model': ema_student.state_dict(),
                    'optim': optim.state_dict(),
                    'sched': sched.state_dict(),
                    'step': step
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, f'ckpt-step{step}.pt'))

            # Evaluate student model
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                net_IS, net_FID, _ = evaluate(student_sampler, student_model, FLAGS.logdir, f'ckpt-step{step}')
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_student, FLAGS.logdir, f'ckpt-step{step}')
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(", ".join(f'{k}:{v:.3f}' for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()

    writer.close()

def eval():
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt_name = 'ckpt-step800000.pt'
    ckpt = torch.load(os.path.join(FLAGS.logdir, ckpt_name))
    model.load_state_dict(ckpt['net_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model, FLAGS.logdir, ckpt_name.split('.')[0])
    print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples.png'),
        nrow=16)

    model.load_state_dict(ckpt['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model, FLAGS.logdir, ckpt_name.split('.')[0])
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, f'{ckpt_name}samples_ema.png'),
        nrow=16)
    
def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    elif FLAGS.eval:
        eval()
    else:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
