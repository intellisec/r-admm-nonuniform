import argparse
import os
import random
import shutil
import time
import warnings
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import logging
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
import json
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

parser.add_argument('--config_file', type=str, default='', help="define config file")
parser.add_argument('--stage', type=str, default='', help="select the pruning stage")
parser.add_argument('--uniform', action='store_true', help="set if uniform pruning is desired")
parser.add_argument('--sparsity_type', type=str, default='', choices=["channel", "weight"], required=True,
                    help="Set sparsity type")
parser.add_argument('--pruning_rate', type=float, choices=[0.01, 0.1, 0.5], required=True, help="Set the pruning rate")
parser.add_argument('--run_id', type=str, default="0", help="Set if different run id is necessary")
parser.add_argument('--resume', action='store_true', help="Resume training")


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

args = parser.parse_args()
config = Config(args)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log_dir = './logs'
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
logger = logging.getLogger()
logger_name = log_dir + f'/resnet50_{args.stage}_{args.sparsity_type}_{args.run_id}_cifar10.log'
logger.addHandler(
    logging.FileHandler(logger_name, "a")
)
logger.info(json.dumps(config.__dict__, indent=4))



def main():

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
    logger.info(f'Total Number of GPUs: {ngpus_per_node}')
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    global best_acc1
    config.gpu = gpu

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))
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
        logger.info("=> using pre-trained model '{}'".format(config.arch))

        model = models.__dict__[config.arch](pretrained=True)
        logger.info(model)
        param_names = []
        module_names = []
        for name, W in model.named_modules():
            module_names.append(name)
        logger.info(module_names)
        for name, W in model.named_parameters():
            param_names.append(name)
        logger.info(param_names)
    else:
        logger.info("=> creating model '{}'".format(config.arch))
        if config.arch == "alexnet_bn":
            model = AlexNet_BN()
            logger.info(model)
            for i, (name, W) in enumerate(model.named_parameters()):
                logger.info(name)
        else:
            # model = ResNet50()
            if len(gpu_list) > 1:
                logger.info("Using multiple gpus")
                model = nn.DataParallel(
                    models.__dict__[config.arch](), gpu_list,
                ).to(device)
                logger.info(model)
            else:
                model = models.__dict__[config.arch]().to(device)
                logger.info(model)

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
        logger.info("GPU not None")
        # torch.cuda.set_device(config.gpu)
        # model = model.cuda(config.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if config.arch.startswith('alexnet') or config.arch.startswith('vgg') or config.arch.startswith('resnet'):
            logger.info("Data Parallel")
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

    # note that loading a pretrain model does not inherit optimizer info
    # will use resume to resume admm training
    if config.load_model:
        radmm_dir = os.getcwd()
        config.load_model = '/'.join(radmm_dir.split('/')[:-3] + ['Pretrain-Models/torch-models', config.load_model])
        if os.path.isfile(config.load_model):
            if not config.admm:
                logger.info(f"Loading state dict from {config.load_model} to device")
                if config.gpu is not None:
                    config.model.load_state_dict(torch.load(config.load_model, map_location=device))
                else:
                    config.model.load_state_dict(torch.load(config.load_model))
            else:
                if config.gpu is not None:
                    if len(gpu_list) > 1:
                        logger.info(f"Loading state dict from {config.load_model} to device")
                        state_dict = torch.load(config.load_model, map_location=device)['state_dict']
                        state_dict = dict([(f"basic_model.{k}", state_dict[k]) for k in state_dict])
                        config.model.load_state_dict(state_dict)
                    else:
                        logger.info(f"Loading state dict from {config.load_model} to device")
                        state_dict = torch.load(config.load_model, map_location=device)['state_dict']
                        state_dict = dict(
                            [(f"basic_model.{k.replace('module.', '')}", state_dict[k]) for k in state_dict])
                        config.model.load_state_dict(state_dict)
                else:
                    logger.info(f"Loading state dict from {config.load_model}")
                    state_dict = torch.load(config.load_model)['state_dict']
                    state_dict = dict([(f"basic_model.{k}", state_dict[k]) for k in state_dict])
                    config.model.load_state_dict(state_dict)
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.load_model))

    config.prepare_pruning()

    admm.predict_sparsity(config)
    # admm.test_sparsity(config)

    ADMM = None

    if config.admm:
        ADMM = admm.ADMM(config)

    config.warmup = (not config.admm) and config.warmup_epochs > 0
    optimizer_init_lr = config.warmup_lr if config.warmup else config.lr

    optimizer = None
    if (config.optimizer == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    elif (config.optimizer == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)

    # optionally resume from a checkpoint
    if config.resume:
        ## will add logic for loading admm variables
        if os.path.isfile(f"checkpoint_{config.save_model}"):
            logger.info("=> loading checkpoint '{}'".format(f"checkpoint_{config.save_model}"))
            checkpoint = torch.load(f"checkpoint_{config.save_model}")
            # config.start_epoch = checkpoint['epoch']

            if config.gpu is not None:
                checkpoint = torch.load(f"checkpoint_{config.save_model}", map_location=device)
                if len(gpu_list) > 1:
                    state_dict = checkpoint['net']
                    # state_dict = dict([(f"basic_model.{k}", state_dict[k]) for k in state_dict])
                    config.model.load_state_dict(state_dict)
                    config.start_epoch = checkpoint['epoch']
                else:
                    state_dict = checkpoint['net']
                    state_dict = dict(
                        [(f"basic_model.{k.replace('module.', '')}", state_dict[k]) for k in state_dict])
                    config.model.load_state_dict(state_dict)
                    config.start_epoch = checkpoint['epoch']
            else:
                checkpoint = torch.load(f"checkpoint_{config.save_model}")
                state_dict = checkpoint['net']
                state_dict = dict([(f"basic_model.{k}", state_dict[k]) for k in state_dict])
                config.model.load_state_dict(state_dict)
                config.start_epoch = checkpoint['epoch']


            global best_mean_loss
            best_mean_loss = checkpoint['best_mean_loss']

            if ADMM:
                ADMM.ADMM_U = checkpoint['admm']['ADMM_U']
                ADMM.ADMM_Z = checkpoint['admm']['ADMM_Z']

            # model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(f"checkpoint_{config.save_model}", checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(f"checkpoint_{config.save_model}"))

    cudnn.benchmark = True

    # Data loading code
    data_dir = '/'.join(os.getcwd().split('/')[:-3] + [config.data])
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None),
        num_workers=8, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    scheduler = None
    if config.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * len(train_loader),
                                                         eta_min=4e-08)
    elif config.lr_scheduler == 'default':
        # sets the learning rate to the initial LR decayed by gamma every 30 epochs"""
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30 * len(train_loader), gamma=0.1)
    else:
        raise Exception("unknown lr scheduler")

    if config.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=config.lr / config.warmup_lr,
                                           total_iter=config.warmup_epochs * len(train_loader),
                                           after_scheduler=scheduler)

    if config.admm:
        validate(val_loader, criterion, config, ADMM, 0, optimizer)

    if config.verify:
        admm.masking(config)
        admm.test_sparsity(config)
        validate(val_loader, criterion, config, ADMM, 0, optimizer)
        import sys
        sys.exit()

    if config.masked_retrain:
        # make sure small weights are pruned and confirm the acc
        admm.masking(config)
        logger.info("before retrain starts")
        admm.test_sparsity(config)
        validate(val_loader, criterion, config, ADMM, 0, optimizer)
    if config.masked_progressive:
        admm.zero_masking(config)

    config.epochs = int(math.ceil(config.epochs / config.n_repeats))
    for epoch in range(config.start_epoch, config.epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, config, ADMM, criterion, optimizer, scheduler, epoch)

        # evaluate on validation set
        validate(val_loader, criterion, config, ADMM, epoch, optimizer)

        # # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)
        #
        # if is_best and not config.admm:  # we don't need admm to have best validation acc
        #     logger.info('saving new best model {}'.format(config.save_model))
        #     torch.save(model.state_dict(), config.save_model)
        #
        # if not config.multiprocessing_distributed or (config.multiprocessing_distributed
        #                                               and config.rank % ngpus_per_node == 0):
        #     save_checkpoint(config, {'admm': {},
        #                              'epoch': epoch + 1,
        #                              'arch': config.arch,
        #                              'state_dict': model.state_dict(),
        #                              'best_acc1': best_acc1,
        #                              'optimizer': optimizer.state_dict(),
        #                              }, is_best)

    # save last model for admm, optimizer detail is not necessary
    if config.save_model and config.admm:
        logger.info('saving final model {}'.format(config.save_model))
        torch.save(config.model.state_dict(), config.save_model)

    # if config.masked_retrain:
    #     logger.info("after masked retrain")
    #     admm.test_sparsity(config)


def fgsm(gradz, step_size):
    return step_size * torch.sign(gradz)


def train(train_loader, config, ADMM, criterion, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    adv_losses = AverageMeter()
    adv_top1 = AverageMeter()
    adv_top5 = AverageMeter()

    # switch to train mode
    config.model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        if config.admm:
            admm.admm_adjust_learning_rate(optimizer, epoch, config)
        else:
            scheduler.step()

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        data = input

        if config.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, config.alpha)

        # compute output with forward
        for _ in range(config.n_repeats):
            adv_output, noise_batch = config.model(input)

            if config.mixup:
                adv_loss = mixup_criterion(criterion, adv_output, target_a, target_b, lam, config.smooth)
            else:
                adv_loss = criterion(adv_output, target, smooth=config.smooth)

            if config.admm:
                admm.admm_update(config, ADMM, device, train_loader, optimizer, epoch, data, i)  # update Z and U
                adv_loss, admm_loss, mixed_loss = admm.append_admm_loss(config, ADMM, adv_loss)  # append admm losss

            adv_acc1, adv_acc5 = accuracy(adv_output, target, topk=(1, 5))

            adv_losses.update(adv_loss.item(), input.size(0))
            adv_top1.update(adv_acc1[0], input.size(0))
            adv_top5.update(adv_acc5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            if config.admm:
                mixed_loss.backward()
            else:
                adv_loss.backward()
            if config.masked_progressive:
                with torch.no_grad():
                    for name, W in config.model.named_parameters():
                        if name in config.zero_masks:
                            W.grad *= config.zero_masks[name]
            if config.masked_retrain:
                with torch.no_grad():
                    for name, W in config.model.named_parameters():
                        if name in config.masks:
                            W.grad *= config.masks[name]

            pert = fgsm(noise_batch.grad, config.epsilon)
            config.model.global_noise_data[0: input.size(0)] += pert.data
            config.model.global_noise_data.clamp_(-config.epsilon, config.epsilon)
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Adv_Loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\t'
                      'Adv_Acc@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})\t'
                      'Adv_Acc@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, adv_loss=adv_losses,
                    adv_top1=adv_top1, adv_top5=adv_top5))


def validate(val_loader, criterion, config, ADMM, epoch, optimizer):
    logger.info("Validating..")
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

        # Validate with PGD
        orig_input = input.clone()
        randn = torch.FloatTensor(input.size()).uniform_(-config.pgd_epsilon, config.pgd_epsilon).to(device)
        input += randn
        input.clamp_(0, 1.0)
        for _ in range(config.pgd_steps):
            invar = Variable(input, requires_grad=True)
            in1 = invar - mean
            in1.div_(std)
            output = config.model.basic_model(in1)
            ascend_loss = criterion(output, target)
            ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
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

    logger.info(' * Nat_Acc@1 {nat_top1.avg:.3f} *Adv_Acc@1 {adv_top1.avg:.3f}'
          .format(nat_top1=nat_top1, adv_top1=adv_top1))

    global best_mean_loss
    global best_adv_acc
    global best_nat_acc
    mean_loss = (adv_losses.avg + nat_losses.avg) / 2
    if mean_loss < best_mean_loss and not config.admm:
        best_mean_loss = mean_loss
        best_adv_acc = adv_top1
        best_nat_acc = nat_top1
        logger.info('new best_mean_loss is {}'.format(mean_loss))
        logger.info('saving model {}'.format(config.save_model))
        """
        torch.save({
            "net": config.model.state_dict(),
            "epoch": (int(epoch)+1)
        },config.save_model)
        """
        torch.save({
            "net": config.model.state_dict()
        }, f"BEST_{config.save_model}")

    if config.save_model and config.admm:
        logger.info('saving checkpoint model checkpoint_{}'.format(config.save_model))
        # torch.save(config.model.state_dict(),config.save_model)
        torch.save({
            "net": config.model.state_dict(),
            "epoch": (int(epoch) + 1),
            "admm": {'ADMM_U': ADMM.ADMM_U, 'ADMM_Z': ADMM.ADMM_Z},
            "best_mean_loss": best_mean_loss,
            "optimizer": optimizer.state_dict()
        }, f"checkpoint_{config.save_model}")

    if config.save_model and not config.admm:
        logger.info('saving checkpoint model checkpoint_{}'.format(config.save_model))
        torch.save({
            "net": config.model.state_dict(),
            "epoch": (int(epoch) + 1),
            "best_mean_loss": best_mean_loss,
            "optimizer": optimizer.state_dict()
        }, f"checkpoint_{config.save_model}")

    return adv_top1.avg


def save_checkpoint(config, state, is_best, filename='checkpoint.pth.tar', ADMM=None):
    filename = 'checkpoint_gpu{}.pth.tar'.format(config.gpu)

    if config.admm and ADMM:  ## add admm variables to state
        state['admm']['ADMM_U'] = ADMM.ADMM_U
        state['admm']['ADMM_Z'] = ADMM.ADMM_Z

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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