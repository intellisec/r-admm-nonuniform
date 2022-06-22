'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import json
import time
import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--config_file', type=str, default='config.yaml', help="config file")
parser.add_argument('--stage', type=str, default='retrain', help="select the pruning stage", choices=['admm','pretrain','retrain'])
parser.add_argument('--arch', type=str, default='vgg16_bn', help="select the model arch", choices=['vgg16_bn', 'resnet18','wrn_28_4'])
parser.add_argument('--uniform', action='store_true', default=False, help="set if uniform pruning is desired")
parser.add_argument('--sparsity_type', type=str, default='weight', choices=["channel", "weight"], help="Set sparsity type")
parser.add_argument('--pruning_rate', type=float, default=0.01, choices=[0.01, 0.1, 0.5], help="Set the pruning rate")
parser.add_argument('--rate_from_config', action='store_true', help="Set if pruning rate from config should be taken")
parser.add_argument('--run_id', type=str, default="uni", help="Set if different run id is necessary")
parser.add_argument('--stg_mode', type=str, default="", help="Source strategy mode")
parser.add_argument('--gpu', type=str, default='0', help="Set gpu id to use")
# init = Init_Func(config.init_func)
# torch.manual_seed(config.random_seed)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import sys
import numpy as np
import logging
from utils import *
from models import *
from config import Config
from sklearn.metrics import balanced_accuracy_score

sys.path.append('../../')  # append root directory

from ADMM_examples.svhn.models.wrn import Wide_ResNet_28_4
from admm.cross_entropy import CrossEntropyLossMaybeSmooth


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AttackPGD(nn.Module):
    def __init__(self, basic_model, config):
        super(AttackPGD, self).__init__()
        self.basic_model = basic_model
        self.rand = config.random_start
        self.step_size = config.step_size / 255
        self.epsilon = config.epsilon / 255
        self.num_steps = 20  # config.num_steps

    def forward(self, input, target):  # do forward in the module.py
        # if not args.attack :
        #    return self.basic_model(input), input

        x = input.detach()

        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_model(x)
                loss = F.cross_entropy(logits, target, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, input - self.epsilon), input + self.epsilon)

            x = torch.clamp(x, 0, 1)

        return self.basic_model(input), self.basic_model(x), x


def fgsm(model, input, target, step_size):
    """
    FGSM training added for FGSM warmup
    """
    x_adv = Variable(input.data, requires_grad=True)
    opt = optim.SGD([x_adv], lr=1e-3)
    opt.zero_grad()

    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(x_adv), target)

    # with torch.enable_grad():
    #     logits = model(x)
    #     loss = F.cross_entropy(logits, target, size_average=False)

    loss.backward()
    eta = step_size * x_adv.grad.data.sign()
    x_adv = Variable(x_adv.data + eta, requires_grad=True)
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=True)

    # grad = torch.autograd.grad(loss, [x])[0]
    #
    # x = x + step_size * torch.sign(grad.data)
    # x = torch.clamp(x, 0, 1)

    return model(input), model(x_adv), x_adv


def cw_loss(output, target,confidence=50, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss


def cw_whitebox(
    model,
    x,
    y,
    epsilon,
    num_steps,
    step_size,
    device,
    clip_min=0.0,
    clip_max=1.0,
    is_random=True,
):

    x_adv = Variable(x.data, requires_grad=True)
    if is_random:
        random_noise = (
            torch.FloatTensor(x_adv.shape).uniform_(-epsilon, epsilon).to(device)
        )
        x_adv = Variable(x_adv.data + random_noise, requires_grad=True)

    for _ in range(num_steps):

        opt = optim.SGD([x_adv], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = cw_loss(model(x_adv), y)
        loss.backward()
        eta = step_size * x_adv.grad.data.sign()
        x_adv = Variable(x_adv.data + eta, requires_grad=True)
        eta = torch.clamp(x_adv.data - x.data, -epsilon, epsilon)
        x_adv = Variable(x.data + eta, requires_grad=True)
        x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=True)

    return model(x), model(x_adv), x_adv

def pgd_whitebox(
    model,
    x,
    y,
    epsilon,
    num_steps,
    step_size,
    device,
    clip_min=0.0,
    clip_max=1.0,
    is_random=True,
):
    x_pgd = Variable(x.data, requires_grad=True)
    if is_random:
        random_noise = (
            torch.FloatTensor(x_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        )
        x_pgd = Variable(x_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([x_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(x_pgd), y)
        loss.backward()
        eta = step_size * x_pgd.grad.data.sign()
        x_pgd = Variable(x_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(x_pgd.data - x.data, -epsilon, epsilon)
        x_pgd = Variable(x.data + eta, requires_grad=True)
        x_pgd = Variable(torch.clamp(x_pgd, clip_min, clip_max), requires_grad=True)

    return model(x), model(x_pgd), x_pgd

ATTACK_LIST = ['fgsm', 'pgd10', 'pgd20', 'cw']

best_nat_acc = AverageMeter()
best_adv_acc = AverageMeter()

config = Config(args)

config.resume = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_mean_loss = 100.  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

str_rate = str(args.pruning_rate).replace('.', '')
if args.sparsity_type == 'weight' and str_rate == '01':
    str_rate = '010'
checkpoint_name = f'{args.arch}_{args.stage}_{args.sparsity_type}_{args.run_id}'
if args.stg_mode != '' and args.stg_mode != 'uni':
    checkpoint_name += f'_{args.stg_mode}'

source_net = f'BEST_{checkpoint_name}.pth.tar'

log_dir = './eval'
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
log_fname = f"adv_bacc_{checkpoint_name}_{args.stg_mode}.log"
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
logger.addHandler(
    logging.FileHandler(os.path.join(log_dir, log_fname), "a")
)
logger.info(args)
logger.info(json.dumps(config.__dict__, indent=4))

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()

])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

data_dir = '/'.join(os.getcwd().split('/')[:-3] + ['data/CIFAR10'])

trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=config.workers)

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=config.workers)

# Model
print('==> Building model..')
model = None
if config.arch == "vgg16_bn":
    model = VGG('vgg16')
elif config.arch == 'vgg16_adv':
    model = VGG_adv('vgg16', w=config.width_multiplier)
elif config.arch == 'vgg16_ori_adv':
    model = VGG_ori_adv('vgg11', w=config.width_multiplier)
elif config.arch == "resnet18":
    model = ResNet18_adv(w=config.width_multiplier)
elif config.arch == "googlenet":
    model = GoogLeNet()
elif config.arch == "densenet121":
    model = DenseNet121()
elif config.arch == "vgg16_1by8":
    model = VGG('vgg16_1by8')
elif config.arch == "vgg16_1by16":
    model = VGG('vgg16_1by16')
elif config.arch == "vgg16_1by32":
    model = VGG('vgg16_1by32')
elif config.arch == "resnet18_1by16":
    model = ResNet18_1by16()
elif config.arch == 'resnet18_adv':
    model = ResNet18_adv(w=config.width_multiplier)
elif config.arch == 'lenet_adv':
    model = LeNet_adv(w=config.width_multiplier)
elif config.arch == 'lenet':
    model = LeNet(w=config.width_multiplier)
elif config.arch == 'resnet18_adv_wide':
    model = ResNet18_adv_wide()
elif config.arch == 'wrn_28_4':
    model = Wide_ResNet_28_4()

print(model)

model = AttackPGD(model, config)
config.model = model

if device == 'cuda':
    if config.gpu is not None:
        torch.cuda.set_device(0)
        config.model = torch.nn.DataParallel(model, device_ids=[0])
    else:
        config.model.cuda()
        config.model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

load_model = source_net

if config.resume:
    # Load checkpoint.
    logger.info(f'==> Resuming from checkpoint: {source_net}')
    # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(load_model, map_location=torch.device(f'cuda:{0}'))
    config.model.load_state_dict(checkpoint['net'])

criterion = CrossEntropyLossMaybeSmooth(smooth_eps=config.smooth_eps).cuda(config.gpu)
config.smooth = config.smooth_eps > 0.0
config.mixup = config.alpha > 0.0

# config.warmup = (not config.admm) and config.warmup_epochs > 0
# Updated definition of warmup to use it already in the admm stage
config.warmup = config.warmup_epochs > 0 and config.admm
optimizer_init_lr = config.warmup_lr if config.warmup else config.lr

optimizer = None
if (config.optimizer == 'sgd'):
    optimizer = torch.optim.SGD(config.model.parameters(), optimizer_init_lr,
                                momentum=0.9,
                                weight_decay=1e-4)
elif (config.optimizer == 'adam'):
    optimizer = torch.optim.Adam(config.model.parameters(), optimizer_init_lr)


def validate(val_loader, criterion, config, logger):

    for attack in ATTACK_LIST:

        batch_time = AverageMeter()
        nat_losses = AverageMeter()
        adv_losses = AverageMeter()
        nat_top1 = AverageMeter()
        adv_top1 = AverageMeter()
        nat_labels = []
        nat_preds_all = []
        adv_preds_all = []

        # switch to evaluate mode
        config.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(val_loader):
                if config.gpu is not None:
                    input = input.cuda(config.gpu, non_blocking=True)
                    target = target.cuda(config.gpu, non_blocking=True)

                # compute output
                if attack == 'fgsm':
                    nat_output, adv_output, _ = fgsm(model.basic_model, input, target, config.epsilon/255.0)
                elif attack == 'cw':
                    nat_output, adv_output, _ = cw_whitebox(model.basic_model, input, target,
                                                            epsilon=config.epsilon/255.0,
                                                            num_steps=20,
                                                            step_size=config.step_size/255.0,
                                                            device=device)
                else:
                    if attack == 'pgd10':
                        nat_output, adv_output, _ = pgd_whitebox(model.basic_model, input, target,
                                                                epsilon=config.epsilon / 255.0,
                                                                num_steps=10,
                                                                step_size=config.step_size / 255.0,
                                                                device=device)

                    if attack == 'pgd20':
                        config.model.num_steps = 20
                        nat_output, adv_output, _ = pgd_whitebox(model.basic_model, input, target,
                                                                 epsilon=config.epsilon / 255.0,
                                                                 num_steps=20,
                                                                 step_size=config.step_size / 255.0,
                                                                 device=device)

                nat_loss = criterion(nat_output, target)
                adv_loss = criterion(adv_output, target)

                # measure accuracy and record loss
                _, nat_preds = nat_output.topk(1, 1, True, True)
                nat_preds = nat_preds.view(-1).cpu().numpy()
                nat_labels = np.append(nat_labels, target.cpu().numpy().squeeze())
                nat_preds_all = np.append(nat_preds_all, nat_preds)

                _, adv_preds = adv_output.topk(1, 1, True, True)
                adv_preds = adv_preds.view(-1).cpu().numpy()
                adv_preds_all = np.append(adv_preds_all, adv_preds)

                # measure accuracy and record loss
                nat_acc1, nat_acc5 = accuracy(nat_output, target, topk=(1, 5))
                adv_acc1, adv_acc5 = accuracy(adv_output, target, topk=(1, 5))
                nat_losses.update(nat_loss.item(), input.size(0))
                adv_losses.update(adv_loss.item(), input.size(0))
                nat_top1.update(nat_acc1[0], input.size(0))
                adv_top1.update(adv_acc1[0], input.size(0))

                top1_bacc = balanced_accuracy_score(nat_labels, nat_preds_all) * 100.0
                adv_top1_bacc = balanced_accuracy_score(nat_labels, adv_preds_all) * 100.0

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Nat_Loss {nat_loss.val:.4f} ({nat_loss.avg:.4f})\t'
                                'Nat_Acc@1 {nat_top1.val:.3f} ({nat_top1.avg:.3f})\t'
                                'Adv_Loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\t'
                                'Adv_Acc@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})\t'
                                .format(
                                i, len(val_loader), batch_time=batch_time, nat_loss=nat_losses,
                                nat_top1=nat_top1, adv_loss=adv_losses, adv_top1=adv_top1))

            print(' * Nat_Acc@1 {nat_top1.avg:.3f} *Adv_Acc@1 {adv_top1.avg:.3f}'.format(nat_top1=nat_top1, adv_top1=adv_top1))

        # logger.info(
        #     f"BALANCED ACC: Benign validation accuracy: {top1_bacc:.2f}, Adversarial validation accuracy by {attack.upper()}: {adv_top1_bacc:.2f}")
        logger.info(
            f"STANDARD ACC: Benign validation accuracy: {nat_top1.avg:.2f}, Adversarial validation accuracy by {attack.upper()}: {adv_top1.avg:.2f}")


if __name__ == '__main__':
    validate(testloader, criterion, config, logger)
