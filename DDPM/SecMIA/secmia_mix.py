
import copy
import sys
import os
import numpy as np
import random
import tqdm
import argparse

from sklearn import metrics
from dataset_utils import load_member_data
from absl import flags
from model import UNet
import torch
import resnet

def get_FLAGS(flag_path):
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
    flags.DEFINE_integer('total_steps', 800000, help='total training steps')
    flags.DEFINE_integer('img_size', 32, help='image size')
    flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
    flags.DEFINE_integer('batch_size', 128, help='batch size')
    flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
    flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
    flags.DEFINE_bool('parallel', False, help='multi gpu training')
    # Logging & Sampling
    flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
    flags.DEFINE_integer('sample_size', 64, help="sampling size of images")
    flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
    # Evaluation
    flags.DEFINE_integer('save_step', 80000, help='frequency of saving checkpoints, 0 to disable during training')
    flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
    flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
    flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
    flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
    # Additional flags
    flags.DEFINE_string('model_dir', 'experiments/CIFAR10', help='directory for model checkpoints')
    flags.DEFINE_string('model_name1', None, help='directory for model checkpoints')
    flags.DEFINE_string('model_name2', None, help='directory for model checkpoints')
    flags.DEFINE_string('dataset', 'CIFAR10', help='data set')
    flags.DEFINE_string('dataset_root', './datasets', help='dataset')
    flags.DEFINE_integer('t_sec', 100, help='number of timesteps for secure MIA')
    flags.DEFINE_integer('k', 10, help='step size for secure MIA')

    FLAGS.read_flags_from_files(flag_path)
    return FLAGS


def ddim_singlestep(model, FLAGS, x, t_c, t_target, requires_grad=False, device='cuda'):

    x = x.to(device)

    t_c = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_c)
    t_target = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_target)

    betas = torch.linspace(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).double().to(device)
    alphas = 1. - betas
    alphas = torch.cumprod(alphas, dim=0)

    alphas_t_c = extract(alphas, t=t_c, x_shape=x.shape)
    alphas_t_target = extract(alphas, t=t_target, x_shape=x.shape)

    if requires_grad:
        epsilon = model(x, t_c)
    else:
        with torch.no_grad():
            epsilon = model(x, t_c)

    pred_x_0 = (x - ((1 - alphas_t_c).sqrt() * epsilon)) / (alphas_t_c.sqrt())
    x_t_target = alphas_t_target.sqrt() * pred_x_0 \
                 + (1 - alphas_t_target).sqrt() * epsilon

    return {
        'x_t_target': x_t_target,
        'epsilon': epsilon
    }


def ddim_multistep(model1, model2, FLAGS, x, t_c, target_steps, clip=False, device='cuda', requires_grad=False):
    for idx, t_target in enumerate(target_steps): 
        if idx == 0:
            result = ddim_singlestep(model1, FLAGS, x, t_c, t_target, requires_grad=requires_grad, device=device)
        else:
            result = ddim_singlestep(model2, FLAGS, x, t_c, t_target, requires_grad=requires_grad, device=device)
        x = result['x_t_target']
        t_c = t_target

    if clip:
        result['x_t_target'] = torch.clip(result['x_t_target'], -1, 1)

    return result


class MIDataset():

    def __init__(self, member_data, nonmember_data, member_label, nonmember_label):
        self.data = torch.concat([member_data, nonmember_data])
        self.label = torch.concat([member_label, nonmember_label]).reshape(-1)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, item):
        data = self.data[item]
        return data, self.label[item]


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_models(ckpt1, ckpt2, device, FLAGS, WA=True):
    model1 = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    model2 = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)

    ckpt1 = torch.load(ckpt1, map_location=device)
    ckpt2 = torch.load(ckpt2, map_location=device)

    weights1 = ckpt1['ema_model'] if WA else ckpt1['net_model']
    weights2 = ckpt2['ema_model'] if WA else ckpt2['net_model']

    model1.load_state_dict(weights1)
    model2.load_state_dict(weights2)

    model1.eval()
    model2.eval()

    return model1, model2


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def norm(x):
    return (x + 1) / 2


def get_intermediate_results(model1, model2, FLAGS, data_loader, t_sec, timestep):

    target_steps = list(range(0, t_sec, timestep))[1:]

    internal_diffusion_list = []
    internal_denoised_list_m1 = []
    internal_denoised_list_m2 = []
    for batch_idx, x in enumerate(tqdm.tqdm(data_loader)):
        x = x[0].cuda()
        x = x * 2 - 1

        x_sec = ddim_multistep(model1, model2, FLAGS, x, t_c=0, target_steps=target_steps)
        x_sec = x_sec['x_t_target']

        x_sec_recon = ddim_singlestep(model1, FLAGS, x_sec, t_c=target_steps[-1], t_target=target_steps[-1] + timestep)
        x_sec_recon = ddim_singlestep(model1, FLAGS, x_sec_recon['x_t_target'], t_c=target_steps[-1] + timestep, t_target=target_steps[-1])
        x_sec_recon1 = x_sec_recon['x_t_target']

        x_sec_recon = ddim_singlestep(model2, FLAGS, x_sec, t_c=target_steps[-1], t_target=target_steps[-1] + timestep)
        x_sec_recon = ddim_singlestep(model2, FLAGS, x_sec_recon['x_t_target'], t_c=target_steps[-1] + timestep, t_target=target_steps[-1])
        x_sec_recon2 = x_sec_recon['x_t_target']

        internal_diffusion_list.append(x_sec)
        internal_denoised_list_m1.append(x_sec_recon1)
        internal_denoised_list_m2.append(x_sec_recon2)

    return {
        'internal_diffusions': torch.concat(internal_diffusion_list),
        'internal_denoise_m1': torch.concat(internal_denoised_list_m1),
        'internal_denoise_m2': torch.concat(internal_denoised_list_m2)
    }


def secmi_attack(model1, model2, FLAGS, dataset_root, timestep=10, t_sec=100, batch_size=128, dataset='cifar10'):
    # load splits
    _, _, member_loader, nonmember_loader = load_member_data(dataset_root=dataset_root, dataset_name=dataset, batch_size=batch_size,
                                                             shuffle=False, randaugment=False)

    member_results = get_intermediate_results(model1, model2, FLAGS, member_loader, t_sec, timestep)
    nonmember_results = get_intermediate_results(model1, model2, FLAGS, nonmember_loader, t_sec, timestep)

    t_results = {
        'member_diffusions': member_results['internal_diffusions'],
        'member_internal_samples_m1': member_results['internal_denoise_m1'],
        'member_internal_samples_m2': member_results['internal_denoise_m2'],
        'nonmember_diffusions': nonmember_results['internal_diffusions'],
        'nonmember_internal_samples_m1': nonmember_results['internal_denoise_m1'],
        'nonmember_internal_samples_m2': nonmember_results['internal_denoise_m2'],
    }


    stat_results_m1, stat_results_m2 = execute_attack(t_results, type='stat')
    print('#' * 20 + ' SecMI_stat ' + '#' * 20)
    print('#' * 20 + ' Model 1 ' + '#' * 20)
    print_result(stat_results_m1)
    print('#' * 20 + ' Model 2 ' + '#' * 20)
    print_result(stat_results_m2)
    # nns_results = execute_attack(t_results, type='nns')
    # print('#' * 20 + ' SecMI_NNs ' + '#' * 20)
    # print_result(nns_results)

def print_result(results):
    keys = ['auc', 'asr', 'TPR@1%FPR', 'TPR@0.1%FPR', 'threshold']
    for k, v in results.items():
        if k in keys:
            print(f'{k}: {v}')

def naive_statistic_attack(t_results, metric='l2'):
    def measure(diffusion, sample, metric, device='cuda'):
        diffusion = diffusion.to(device).float()
        sample = sample.to(device).float()

        if len(diffusion.shape) == 5:
            num_timestep = diffusion.size(0)
            diffusion = diffusion.permute(1, 0, 2, 3, 4).reshape(-1, num_timestep * 3, 32, 32)
            sample = sample.permute(1, 0, 2, 3, 4).reshape(-1, num_timestep * 3, 32, 32)

        if metric == 'l2':
            score = ((diffusion - sample) ** 2).flatten(1).sum(dim=-1)
        else:
            raise NotImplementedError

        return score

    # member scores
    member_scores_m1 = measure(t_results['member_diffusions'], t_results['member_internal_samples_m1'], metric=metric)
    member_scores_m2 = measure(t_results['member_diffusions'], t_results['member_internal_samples_m2'], metric=metric)
    # nonmember scores
    nonmember_scores_m1 = measure(t_results['nonmember_diffusions'], t_results['nonmember_internal_samples_m1'],
                               metric=metric)
    nonmember_scores_m2 = measure(t_results['nonmember_diffusions'], t_results['nonmember_internal_samples_m2'],
                               metric=metric)
    return member_scores_m1, member_scores_m2, nonmember_scores_m1, nonmember_scores_m2


def execute_attack(t_result, type):
    if type == 'stat':
        member_scores_m1, member_scores_m2, nonmember_scores_m1, nonmember_scores_m2 = naive_statistic_attack(t_result, metric='l2')
    # elif type == 'nns':
    #     member_scores, nonmember_scores, model = nns_attack(t_result, train_portion=0.2)
    #     member_scores *= -1
    #     nonmember_scores *= -1
    else:
        raise NotImplementedError

    auc, asr, fpr_list, tpr_list, threshold = roc(member_scores_m1, nonmember_scores_m1, n_points=2000)
    # TPR @ 1% FPR
    tpr_1_fpr = tpr_list[(fpr_list - 0.01).abs().argmin(dim=0)]
    # TPR @ 0.1% FPR
    tpr_01_fpr = tpr_list[(fpr_list - 0.001).abs().argmin(dim=0)]

    exp_data_m1 = {
        'member_scores': member_scores_m1,  # for histogram
        'nonmember_scores': nonmember_scores_m1,
        'asr': asr.item(),
        'auc': auc,
        'fpr_list': fpr_list,
        'tpr_list': tpr_list,
        'TPR@1%FPR': tpr_1_fpr,
        'TPR@0.1%FPR': tpr_01_fpr,
        'threshold': threshold
    }

    auc, asr, fpr_list, tpr_list, threshold = roc(member_scores_m2, nonmember_scores_m2, n_points=2000)
    # TPR @ 1% FPR
    tpr_1_fpr = tpr_list[(fpr_list - 0.01).abs().argmin(dim=0)]
    # TPR @ 0.1% FPR
    tpr_01_fpr = tpr_list[(fpr_list - 0.001).abs().argmin(dim=0)]

    exp_data_m2 = {
        'member_scores': member_scores_m2,  # for histogram
        'nonmember_scores': nonmember_scores_m2,
        'asr': asr.item(),
        'auc': auc,
        'fpr_list': fpr_list,
        'tpr_list': tpr_list,
        'TPR@1%FPR': tpr_1_fpr,
        'TPR@0.1%FPR': tpr_01_fpr,
        'threshold': threshold
    }

    return exp_data_m1, exp_data_m2


def roc(member_scores, nonmember_scores, n_points=1000):
    max_asr = 0
    max_threshold = 0

    min_conf = min(member_scores.min(), nonmember_scores.min()).item()
    max_conf = max(member_scores.max(), nonmember_scores.max()).item()

    FPR_list = []
    TPR_list = []

    for threshold in torch.arange(min_conf, max_conf, (max_conf - min_conf) / n_points):
        TP = (member_scores <= threshold).sum()
        TN = (nonmember_scores > threshold).sum()
        FP = (nonmember_scores <= threshold).sum()
        FN = (member_scores > threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        ASR = (TP + TN) / (TP + TN + FP + FN)

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        if ASR > max_asr:
            max_asr = ASR
            max_threshold = threshold

    FPR_list = np.asarray(FPR_list)
    TPR_list = np.asarray(TPR_list)
    auc = metrics.auc(FPR_list, TPR_list)
    return auc, max_asr, torch.from_numpy(FPR_list), torch.from_numpy(TPR_list), max_threshold


def nns_attack(t_results, train_portion=0.5, device='cuda'):
    n_epoch = 15
    lr = 0.001
    batch_size = 128
    # model training
    train_loader, test_loader, num_timestep = split_nn_datasets(t_results, train_portion=train_portion,
                                                                batch_size=batch_size)
    print(f'num timestep: {num_timestep}')
    # initialize NNs
    model = resnet.ResNet18(num_channels=3 * num_timestep * 1, num_classes=1).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # model eval

    test_acc_best_ckpt = None
    test_acc_best = 0
    for epoch in range(n_epoch):
        train_loss, train_acc = nn_train(epoch, model, optim, train_loader)
        test_loss, test_acc = nn_eval(model, test_loader)
        if test_acc > test_acc_best:
            test_acc_best_ckpt = copy.deepcopy(model.state_dict())

    # resume best ckpt
    model.load_state_dict(test_acc_best_ckpt)
    model.eval()
    # generate member_scores, nonmember_scores
    member_scores = []
    nonmember_scores = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            logits = model(data.to(device))
            member_scores.append(logits[label == 1])
            nonmember_scores.append(logits[label == 0])

    member_scores = torch.concat(member_scores).reshape(-1)
    nonmember_scores = torch.concat(nonmember_scores).reshape(-1)
    return member_scores, nonmember_scores, model


def nn_train(epoch, model, optimizer, data_loader, device='cuda'):
    model.train()

    mean_loss = 0
    total = 0
    acc = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data = data.to(device)
        label = label.to(device).reshape(-1, 1)

        logit = model(data)

        loss = ((logit - label) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
        total += data.size(0)

        logit[logit >= 0.5] = 1
        logit[logit < 0.5] = 0
        acc += (logit == label).sum()

    mean_loss /= len(data_loader)
    print(f'Epoch: {epoch} \t Loss: {mean_loss:.4f} \t Acc: {acc / total:.4f} \t')
    return mean_loss, acc / total


def split_nn_datasets(t_results, train_portion=0.1, batch_size=128):
    # split training and testing
    # [t, 25000, 3, 32, 32]
    member_diffusion = t_results['member_diffusions']
    member_sample = t_results['member_internal_samples']
    nonmember_diffusion = t_results['nonmember_diffusions']
    nonmember_sample = t_results['nonmember_internal_samples']
    if len(member_diffusion.shape) == 4:
        # with one timestep
        # minus
        num_timestep = 1
        member_concat = (member_diffusion - member_sample).abs() ** 1
        nonmember_concat = (nonmember_diffusion - nonmember_sample).abs() ** 1
    elif len(member_diffusion.shape) == 5:
        # with multiple timestep
        # minus
        num_timestep = member_diffusion.size(0)
        member_concat = ((member_diffusion - member_sample).abs() ** 2).permute(1, 0, 2, 3, 4).reshape(-1,
                                                                                                       num_timestep * 3,
                                                                                                       32, 32)
        nonmember_concat = ((nonmember_diffusion - nonmember_sample).abs() ** 2).permute(1, 0, 2, 3, 4).reshape(-1,
                                                                                                                num_timestep * 3,
                                                                                                                32, 32)
    else:
        raise NotImplementedError

    # train num
    num_train = int(member_concat.size(0) * train_portion)
    # split
    train_member_concat = member_concat[:num_train]
    train_member_label = torch.ones(train_member_concat.size(0))
    train_nonmember_concat = nonmember_concat[:num_train]
    train_nonmember_label = torch.zeros(train_nonmember_concat.size(0))
    test_member_concat = member_concat[num_train:]
    test_member_label = torch.ones(test_member_concat.size(0))
    test_nonmember_concat = nonmember_concat[num_train:]
    test_nonmember_label = torch.zeros(test_nonmember_concat.size(0))

    # datasets
    if num_train == 0:
        train_dataset = None
        train_loader = None
    else:
        train_dataset = MIDataset(train_member_concat, train_nonmember_concat, train_member_label,
                                  train_nonmember_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MIDataset(test_member_concat, test_nonmember_concat, test_member_label, test_nonmember_label)
    # dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, num_timestep


@torch.no_grad()
def nn_eval(model, data_loader, device='cuda'):
    model.eval()

    mean_loss = 0
    total = 0
    acc = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data, label = data.to(device), label.to(device).reshape(-1, 1)
        logit = model(data)

        loss = ((logit - label) ** 2).mean()

        mean_loss += loss.item()
        total += data.size(0)

        logit[logit >= 0.5] = 1
        logit[logit < 0.5] = 0

        acc += (logit == label).sum()

    mean_loss /= len(data_loader)
    print(f'Test: \t Loss: {mean_loss:.4f} \t Acc: {acc / total:.4f} \t')
    return mean_loss, acc / total


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='experiments/CIFAR10-dual')
    parser.add_argument('--model_name1', type=str, default='ckpt-step240000_model_0.pt')
    parser.add_argument('--model_name2', type=str, default='ckpt-step240000_model_1.pt')
    parser.add_argument('--dataset_root', type=str, default='datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--t_sec', type=int, default=100)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    fix_seed(0)
    ckpt1 = os.path.join(args.model_dir, args.model_name1)
    ckpt2 = os.path.join(args.model_dir, args.model_name2)
    flag_path = os.path.join(args.model_dir, 'flagfile.txt')
    device = 'cuda'
    FLAGS = get_FLAGS(flag_path)
    FLAGS(sys.argv)
    model1, model2 = get_models(ckpt1, ckpt2, device, FLAGS, WA=True)
    model1 = model1.to(device)
    model2 = model2.to(device)
    secmi_attack(model1, model2, FLAGS, dataset_root=args.dataset_root, t_sec=args.t_sec, timestep=args.k, batch_size=1024, dataset=args.dataset)
