import argparse
import os
import random
import json
import logging
import time
import warnings
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import numpy as np
import math
from numpy import linalg as LA
import models
from config import Config
from torch.autograd import Variable

sys.path.append('../../')  # append root directory

from admm.warmup_scheduler import GradualWarmupScheduler
from admm.cross_entropy import CrossEntropyLossMaybeSmooth
from admm.utils import mixup_data, mixup_criterion
import admm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--config_file', type=str, default='config.yaml', help="define config file")
parser.add_argument('--stage', type=str, default='retrain', help="select the pruning stage")
parser.add_argument('--uniform', action='store_true', help="set if uniform pruning is desired")
parser.add_argument('--sparsity_type', type=str, default='channel', choices=["channel", "weight"],
                    help="Set sparsity type")
parser.add_argument('--pruning_rate', type=float, default=0.5, choices=[0.01, 0.1, 0.5], help="Set the pruning rate")
parser.add_argument('--run_id', type=str, default="0", help="Set if different run id is necessary")
parser.add_argument('--resume', action='store_true', default=True, help="Resume training")


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


class FreeAT(nn.Module):
    def __init__(self, basic_model, config):
        super(FreeAT, self).__init__()
        self.basic_model = basic_model
        # check if list is correct type
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.batch_size = config.batch_size
        self.image_dim = 224
        self.global_noise_data = torch.zeros([self.batch_size, 3, self.image_dim, self.image_dim]).to(device)
        self.mean = torch.Tensor(np.array(self.mean)[:, np.newaxis, np.newaxis])
        self.mean = self.mean.expand(3, self.image_dim, self.image_dim).to(device)
        self.std = torch.Tensor(np.array(self.std)[:, np.newaxis, np.newaxis])
        self.std = self.std.expand(3, self.image_dim, self.image_dim).to(device)

    def forward(self, input):
        noise_batch = Variable(
            self.global_noise_data[0: input.size(0)], requires_grad=True
        ).to(device)
        in1 = input + noise_batch
        in1.clamp_(0, 1.0)
        in1.sub_(self.mean).div_(self.std)
        return self.basic_model(in1), noise_batch


best_mean_loss = 100.
best_nat_acc = AverageMeter()
best_adv_acc = AverageMeter()


def main():
    args = parser.parse_args()
    config = Config(args)

    str_rate = str(args.pruning_rate).replace('.', '')
    if args.sparsity_type == 'weight' and str_rate == '01':
        str_rate = '010'
    checkpoint_name = f'resnet50_{args.stage}_{args.sparsity_type}_{str_rate}_{args.run_id}'

    source_net = f'BEST_{checkpoint_name}.pth.tar'

    log_dir = './eval'
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_fname = f"adv_bacc_{checkpoint_name}.log"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(log_dir, log_fname), "a")
    )
    logger.info(args)
    logger.info(json.dumps(config.__dict__, indent=4))

    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print('Total Number of GPUs: ', ngpus_per_node)
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config, source_net, logger)


def main_worker(gpu, ngpus_per_node, config, source_net, logger):
    config.gpu = gpu

    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))
        gpu_list = [int(i) for i in str(config.gpu).strip().split(",")]
        device = torch.device(f"cuda:{gpu_list[0]}")

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)
    # create model
    if config.pretrained:
        print("=> using pre-trained model '{}'".format(config.arch))

        model = models.__dict__[config.arch](pretrained=True)
        print(model)
        param_names = []
        module_names = []
        for name, W in model.named_modules():
            module_names.append(name)
        print(module_names)
        for name, W in model.named_parameters():
            param_names.append(name)
        print(param_names)
    else:
        print("=> creating model '{}'".format(config.arch))
        if config.arch == "alexnet_bn":
            model = AlexNet_BN()
            print(model)
            for i, (name, W) in enumerate(model.named_parameters()):
                print(name)
        else:
            # model = ResNet50()
            if len(gpu_list) > 1:
                print("Using multiple gpus")
                model = nn.DataParallel(
                    models.__dict__[config.arch](), gpu_list,
                ).to(device)
                print(model)
            else:
                model = models.__dict__[config.arch]().to(device)
                print(model)

    if config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu[0])
            model.cuda(config.gpu[0])
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int(config.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_list])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.gpu is not None:
        print("GPU not None")
        # torch.cuda.set_device(config.gpu)
        # model = model.cuda(config.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if config.arch.startswith('alexnet') or config.arch.startswith('vgg') or config.arch.startswith('resnet'):
            print("Data Parallel")
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    config.model = FreeAT(model, config)
    # define loss function (criterion) and optimizer

    # criterion = CrossEntropyLossMaybeSmooth(smooth_eps=config.smooth_eps).cuda(config.gpu)
    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=config.smooth_eps)

    config.smooth = config.smooth_eps > 0.0
    config.mixup = config.alpha > 0.0

    config.prepare_pruning()

    if os.path.isfile(source_net):
        if config.gpu is not None:
            checkpoint = torch.load(source_net, map_location=device)
            if len(gpu_list) > 1:
                state_dict = checkpoint['net']
                # state_dict = dict([(f"basic_model.{k}", state_dict[k]) for k in state_dict])
                config.model.load_state_dict(state_dict)
            else:
                state_dict = checkpoint['net']
                state_dict = dict(
                    [(f"basic_model.{k.replace('module.', '')}", state_dict[k]) for k in state_dict])
                config.model.load_state_dict(state_dict)
        else:
            checkpoint = torch.load(source_net)
            state_dict = checkpoint['net']
            state_dict = dict([(f"basic_model.{k}", state_dict[k]) for k in state_dict])
            config.model.load_state_dict(state_dict)

        # model.load_state_dict(state_dict)
        print("=> loaded checkpoint '{}'".format(source_net))
    else:
        print("=> no checkpoint found at '{}'".format(source_net))

    cudnn.benchmark = True

    # Data loading code
    data_dir = '/'.join(os.getcwd().split('/')[:-3] + [config.data])
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True)
    for attack in ['fgsm', 'pgd', 'cw']:
        validate(val_loader, criterion, config, logger, attack=attack)

    # if config.masked_retrain:
    #     print("after masked retrain")
    #     admm.test_sparsity(config)


def fgsm(gradz, step_size):
    return step_size * torch.sign(gradz)


def cw_loss(output, target,confidence=50, num_classes=1000):
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


def validate(val_loader, criterion, config, logger, attack='cw'):
    print("Validating..")
    # Mean/Std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_dim = 224
    mean = torch.Tensor(np.array(mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, image_dim, image_dim).to(device)
    std = torch.Tensor(np.array(std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, image_dim, image_dim).to(device)

    batch_time = AverageMeter()
    nat_losses = AverageMeter()
    adv_losses = AverageMeter()
    nat_top1 = AverageMeter()
    adv_top1 = AverageMeter()
    nat_top5 = AverageMeter()
    adv_top5 = AverageMeter()
    # switch to evaluate mode
    config.model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Validate with Adversarial Attacks
        orig_input = input.clone()

        if attack in ['pgd', 'cw']:
            randn = torch.FloatTensor(input.size()).uniform_(-config.pgd_epsilon, config.pgd_epsilon).to(device)
            input += randn
            input.clamp_(0, 1.0)

        for _ in range(config.pgd_steps):
            invar = Variable(input, requires_grad=True)
            in1 = invar - mean
            in1.div_(std)
            output = config.model.basic_model(in1)
            if attack == 'cw':
                with torch.enable_grad():
                    ascend_loss = cw_loss(output, target)
            else:
                ascend_loss = criterion(output, target)

            ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]

            if attack == 'fgsm':
                pert = fgsm(ascend_grad, config.pgd_epsilon)
            else:
                pert = fgsm(ascend_grad, config.pgd_step_size)

            # Apply purturbation
            input += pert.data
            input = torch.max(orig_input - config.pgd_epsilon, input)
            input = torch.min(orig_input + config.pgd_epsilon, input)
            input.clamp_(0, 1.0)

        input.sub_(mean).div_(std)
        orig_input.sub_(mean).div_(std)
        with torch.no_grad():
            nat_output = config.model.basic_model(orig_input)
            adv_output = config.model.basic_model(input)

            nat_loss = criterion(nat_output, target)
            adv_loss = criterion(adv_output, target)

            # measure accuracy and record loss
            nat_acc1, nat_acc5 = accuracy(nat_output, target, topk=(1, 5))
            adv_acc1, adv_acc5 = accuracy(adv_output, target, topk=(1, 5))
            nat_losses.update(nat_loss.item(), input.size(0))
            adv_losses.update(adv_loss.item(), input.size(0))
            nat_top1.update(nat_acc1[0], input.size(0))
            adv_top1.update(adv_acc1[0], input.size(0))
            nat_top5.update(nat_acc5[0], input.size(0))
            adv_top5.update(adv_acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % config.print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Nat_Loss {nat_loss.val:.4f} ({nat_loss.avg:.4f})\t'
                  'Nat_Acc@1 {nat_top1.val:.3f} ({nat_top1.avg:.3f})\t'
                  'Nat_Acc@5 {nat_top5.val:.3f} ({nat_top5.avg:.3f})\t'
                  'Adv_Loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\t'
                  'Adv_Acc@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})\t'
                  'Adv_Acc@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})\t'
                .format(
                i, len(val_loader), batch_time=batch_time, nat_loss=nat_losses,
                nat_top1=nat_top1, nat_top5=nat_top5, adv_loss=adv_losses, adv_top1=adv_top1, adv_top5=adv_top5))

    logger.info(f' * Nat_Acc@1 {nat_top1.avg:.2f} *Adv_Acc@1 by {attack.upper()}: {adv_top1.avg:.2f}')

    return adv_top1.avg


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


if __name__ == '__main__':
    main()