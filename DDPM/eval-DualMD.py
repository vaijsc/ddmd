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
from diffusion import extract
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
flags.DEFINE_string('ckpt1', './experiments/CIFAR10-dual/ckpt-step800000_model_0.pt', help='teacher checkpoint')
flags.DEFINE_string('ckpt2', './experiments/CIFAR10-dual/ckpt-step800000_model_1.pt', help='teacher checkpoint')

flags.DEFINE_integer('num_images', 25000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

device = torch.device('cuda:0')

class GaussianDiffusionSampler(torch.nn.Module):
    def __init__(self, model1, model2, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon', 'eps_xt_xt-1']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model1 = model1
        self.model2 = model2
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

        self.register_buffer(
            'sqrt_recip_alphas',
            1. / torch.sqrt(alphas)
        )
        self.register_buffer(
            'eps_coef',
            self.betas / torch.sqrt(1 - alphas_bar)
        )

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        # 1 / sqrt(alpha^{bar}) * x_t - 1 / sqrt(alpha^{bar}) - 1 * eps
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if t[0] % 2 == 0:
            model_pred = self.model2(x_t, t)
        else:
            model_pred = self.model1(x_t, t)
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=model_pred)
            model_mean = model_pred
        elif self.mean_type == 'xstart':    # the model predicts x_0
            model_mean, _ = self.q_mean_variance(model_pred, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=model_pred)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'eps_xt_xt-1':
            model_mean, _ = self.p_prev_eps_xt(model_pred, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)

        if self.mean_type != 'eps_xt_xt-1':
            x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def p_prev_eps_xt(self, eps, x_t, t):
        sqrt_recip_alphas = extract(self.sqrt_recip_alphas, t=t, x_shape=x_t.shape)
        eps_coef = extract(self.eps_coef, t=t, x_shape=x_t.shape)
        result = sqrt_recip_alphas * (x_t - eps_coef * eps)
        return result, None


    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

def generate_images(model1, model2, x_T, save_name):
    sampler = GaussianDiffusionSampler(model1, model2, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
                                               FLAGS.mean_type, FLAGS.var_type).to(device)
    with torch.no_grad():
        sample_images = sampler(x_T)
        grid = (make_grid(sample_images) + 1) / 2
        save_image(grid, os.path.join(FLAGS.logdir, save_name))

def evaluate(sampler, model1, model2, save_path, save_name):
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

def eval(model1, model2, exp_name):
    sampler = GaussianDiffusionSampler(model1, model2, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
                                               FLAGS.mean_type, FLAGS.var_type).to(device)
    (IS, IS_std), FID, samples = evaluate(sampler, model1, model2, FLAGS.logdir, "mix_inference")
    print("Eval result %s: IS:%6.3f(%.3f), FID:%7.3f" % (exp_name, IS, IS_std, FID))
    # save sampled images
    torch.save({'samples': samples, 'IS': IS, 'IS_std': IS_std, 'FID': FID}, os.path.join(FLAGS.logdir, f'{exp_name}_eval.pt'))

def main(argv):
    os.makedirs(os.path.join(FLAGS.logdir), exist_ok=True)

    # Teacher Model (Pre-trained)
    model1 = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
    model2 = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
    
    # Load pre-trained teacher weights
    ckpt1 = torch.load(FLAGS.ckpt1, map_location=device)
    model1.load_state_dict(ckpt1['net_model'])
    model1.eval()
    model1.requires_grad = False
    ckpt2 = torch.load(FLAGS.ckpt2, map_location=device)
    model2.load_state_dict(ckpt2['net_model'])
    model2.eval()
    model2.requires_grad = False

    x_T = torch.randn(FLAGS.batch_size//32, 3, FLAGS.img_size, FLAGS.img_size).to(device)

    generate_images(model1, model2, x_T, f'mix_gen.png')
    # print("Generated mixed images")
    print(f"Evaluating mixed inference model {FLAGS.ckpt1} and {FLAGS.ckpt2}")
    eval(model1, model2, "mix_inference")

    # generate_images(model1, model1, x_T, f'model1_gen.png')
    # print(f"Evaluating model1 {FLAGS.ckpt1}")
    # eval(model1, model1, "model1")

    # generate_images(model2, model2, x_T, f'model2_gen.png')
    # print(f"Evaluating model2 {FLAGS.ckpt2}")
    # eval(model2, model2, "model2")

if __name__ == '__main__':
    app.run(main)
