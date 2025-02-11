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
from tqdm import trange, tqdm
from diffusion import extract, GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score


FLAGS = flags.FLAGS
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')

flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
flags.DEFINE_string('logdir', './experiments/CIFAR10-dual', help='log directory')

flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_string('ckpt', './experiments/CIFAR10-dual/ckpt-step800000.pt', help='teacher checkpoint')

flags.DEFINE_integer('num_images', 25000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

device = torch.device('cuda:0')

def evaluate(sampler, model, save_path, save_name):
    with torch.no_grad():
        images = []
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc="generating images"):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    (IS, IS_std), FID = get_inception_and_fid_score(images, FLAGS.fid_cache, num_images=FLAGS.num_images,
                                                    use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images

def eval(model, exp_name):
    sampler = GaussianDiffusionSampler(model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
                                               FLAGS.mean_type, FLAGS.var_type).to(device)
    (IS, IS_std), FID, samples = evaluate(sampler, model, FLAGS.logdir, exp_name)
    print("Eval result %s: IS:%6.3f(%.3f), FID:%7.3f" % (exp_name, IS, IS_std, FID))
    # save sampled images
    torch.save({'samples': samples, 'IS': IS, 'IS_std': IS_std, 'FID': FID}, os.path.join(FLAGS.logdir, f'{exp_name}_eval.pt'))

def main(argv):
    os.makedirs(os.path.join(FLAGS.logdir), exist_ok=True)

    # Teacher Model (Pre-trained)
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
    
    # Load pre-trained teacher weights
    ckpt = torch.load(FLAGS.ckpt, map_location=device)
    model.load_state_dict(ckpt['net_model'])
    model.eval()
    model.requires_grad = False

    print(f"Evaluating model {FLAGS.ckpt}")
    eval(model, "model")


if __name__ == '__main__':
    app.run(main)
