'''configuration function:
'''
import json
import os
import numpy as np
import yaml
import torch.nn as nn


class Config:
    def __init__(self, args):
        """
        Important:
        1. pretrained model has to be named: resnet50_pretrained.pth.tar OR change line 36 to the correpsonding name
        """
        config_dir = args.config_file
        stage_choices = ['admm', 'pretrain', 'retrain']
        stage = args.stage
        # own adaptation
        self.stage = args.stage
        self.arch = "ResNet50"

        if args.uniform:
            self.uniform = True
        else:
            self.uniform = False

        self.sparsity_type = args.sparsity_type
        self.pruning_rate = args.pruning_rate
        self.run_id = args.run_id
        # self.strategy_id = self.run_id
        # self.strategy_id = f"{str(self.pruning_rate).replace('.', '')}_{self.strategy_id}"
        #
        # if self.pruning_rate == 0.1 and self.sparsity_type == "weight":
        #     self.strategy_id = self.strategy_id.replace("_", "0_")

        self.save_model = f"resnet50_{self.stage}_{self.sparsity_type}_{self.pruning_rate}_{self.run_id}.pth.tar"

        if self.stage == 'retrain':
            self.load_model = f"resnet50_admm_{self.sparsity_type}_{self.pruning_rate}_{self.run_id}.pth.tar"
        elif self.stage == 'admm':
            self.load_model = "resnet50_pretrain_imagenet.pth.tar"
        else:
            self.load_model = None

        print(f"Stage is: {self.stage}")
        print(f"Arch is: Resnet50")
        print(f"Uniform pruning: {self.uniform}")
        print(f"Sparsity type: {self.sparsity_type}")
        print(f"Pruning rate: {self.pruning_rate}")
        print(f"Load model: {self.load_model}")
        print(f"Save model: {self.save_model}")

        ds_name = 'imagenet'
        radmm_dir = os.getcwd()
        strategy_f = '/'.join(radmm_dir.split('/')[:-3] + ['strategies/{}.json'.format(ds_name)])
        strategies = json.load(open(strategy_f))
        heracles_stg = strategies[self.arch][self.sparsity_type][self.run_id]

        self.prune_stg = [[], [], []]  # [[conv_prates], [fc_prates], [shortcut-prates]]
        if self.sparsity_type == 'channel':
            assert len(heracles_stg) == 50, '! ! {}-{} is invalid'.format("ResNet50", self.sparsity_type)
            self.prune_stg[0] = heracles_stg[:49]
            self.prune_stg[1] = heracles_stg[49:]
            self.prune_stg[2] = list(np.take(heracles_stg, (1, 10, 22, 40)))
        if self.sparsity_type == 'weight':
            assert len(heracles_stg) == 54, '! ! {}-{} is invalid'.format("ResNet50", self.sparsity_type)
            self.prune_stg[0] = heracles_stg[:1] + heracles_stg[2:11] + heracles_stg[12:24] + heracles_stg[25:42] + heracles_stg[42:53]
            self.prune_stg[1] = heracles_stg[53:]
            self.prune_stg[2] = list(np.take(heracles_stg, (1, 11, 24, 42)))

        if self.sparsity_type == "weight":
            self.sparsity_type = "irregular"

        try:
            with open(config_dir, "r") as stream:
                raw_dict = yaml.load(stream)

                # pgd
                self.pgd_random_start = raw_dict['general']['pgd_random_start']
                self.pgd_epsilon = raw_dict['general']['pgd_epsilon']
                print(f"Epsilon: {self.pgd_epsilon}")
                self.pgd_steps = raw_dict['general']['pgd_steps']
                self.pgd_step_size = raw_dict['general']['pgd_step_size']
                print(f"Step size: {self.pgd_step_size}")
                self.epsilon = raw_dict['general']['epsilon']
                self.n_repeats = raw_dict['general']['n_repeats']
                # general
                self.print_freq = raw_dict['general']['print_freq']
                self.resume = raw_dict['general']['resume']
                self.resume = args.resume
                self.gpu = raw_dict['general']['gpu']
                if self.gpu == 'all':
                    self.gpu = None
                self.arch = raw_dict['general']['arch']
                self.workers = raw_dict['general']['workers']
                self.logging = raw_dict['general']['logging']
                self.log_dir = raw_dict['general']['log_dir']
                self.smooth_eps = raw_dict['general']['smooth_eps']
                self.alpha = raw_dict['general']['alpha']
                # self.sparsity_type = raw_dict['general']['sparsity_type']
                self.data = raw_dict['general']['data']
                self.batch_size = int(raw_dict['general']['batch_size'])
                self.start_epoch = raw_dict['general']['start_epoch']
                self.pretrained = raw_dict['general']['pretrained']
                self.momentum = float(raw_dict['general']['momentum'])
                self.weight_decay = float(raw_dict['general']['weight_decay'])
                self.world_size = raw_dict['general']['world_size']
                self.rank = raw_dict['general']['rank']
                self.dist_url = raw_dict['general']['dist_url']
                self.dist_backend = raw_dict['general']['dist_backend']
                self.seed = raw_dict['general']['seed']
                self.multiprocessing_distributed = raw_dict['general']['multiprocessing_distributed']
                self.verify = raw_dict['general']['verify']
                if stage != 'pretrain':
                    self._prune_ratios = raw_dict[self.arch]['prune_ratios']
                    if self.uniform:
                        for k, v in self._prune_ratios.copy().items():
                            self._prune_ratios[k] = 1 - self.pruning_rate
                    else:
                        self.insert_nonuniform_strategy()
                self.name_encoder = {}
                self.lr = raw_dict[stage]['lr']
                self.lr_scheduler = raw_dict[stage]['lr_scheduler']
                self.optimizer = raw_dict[stage]['optimizer']
                # self.save_model = raw_dict[stage]['save_model']
                # self.load_model = None  # otherwise key error
                self.masked_progressive = None
                if stage != 'pretrain':
                    # self.load_model = raw_dict[stage]['load_model']
                    self.masked_progressive = raw_dict[stage]['masked_progressive']
                self.epochs = raw_dict[stage]['epochs']

                if stage != 'admm':
                    self.warmup_epochs = raw_dict[stage]['warmup_epochs']
                    self.warmup_lr = raw_dict[stage]['warmup_lr']

                self.admm = (stage == 'admm')
                self.masked_retrain = (stage == 'retrain')
                self.rho = None
                self.rhos = {}
                if stage == 'admm':
                    # admm_pruning
                    self.admm_epoch = raw_dict[stage]['admm_epoch']
                    self.rho = raw_dict[stage]['rho']
                    self.multi_rho = raw_dict[stage]['multi_rho']
                    self.verbose = raw_dict[stage]['verbose']
                # following variables assist the pruning algorithm

                self.masks = None
                self.zero_masks = None
                self.conv_names = []
                self.bn_names = []
                self.fc_names = []
                self.prune_ratios = {}
                self.model = None

        except yaml.YAMLError as exc:
            print(exc)

    def insert_nonuniform_strategy(self):
        for i, (k, v) in enumerate(self._prune_ratios.copy().items()):
            assert len(self._prune_ratios) == 54

            if "fc" in k:
                self._prune_ratios[k] = float(1 - self.prune_stg[1].pop(0))
            elif i in [4, 14, 27, 46]:
                self._prune_ratios[k] = float(1 - self.prune_stg[2].pop(0))
            else:
                self._prune_ratios[k] = float(1 - self.prune_stg[0].pop(0))

    def prepare_pruning(self):

        self._extract_layer_names(self.model)
        for good_name, ratio in self._prune_ratios.items():
            self._encode(good_name)
        for good_name, ratio in self._prune_ratios.items():
            self.prune_ratios[self.name_encoder[good_name]] = ratio
        for k in self.prune_ratios.keys():
            self.rhos[k] = self.rho  # this version we assume all rhos are equal
        print('<========={} conv names'.format(len(self.conv_names)))
        print(self.conv_names)
        print('<========={} bn names'.format(len(self.bn_names)))
        print(self.bn_names)
        print('<========={} targeted pruned layers'.format(len(self.prune_ratios)))
        print(self.prune_ratios.keys())
        for k, v in self.prune_ratios.items():
            print('target sparsity in {} is {}'.format(k, v))

    def __str__(self):
        return str(self.__dict__)

    def _extract_layer_names(self, model):
        """
        Store layer name of different types in arrays for indexing
        """

        names = []
        for name, W in self.model.named_modules():
            names.append(name)
        print(names)
        for name, W in self.model.named_modules():
            name += '.weight'  # name in named_modules looks like module.features.0. We add .weight into it
            if isinstance(W, nn.Conv2d):
                self.conv_names.append(name)
            if isinstance(W, nn.BatchNorm2d):
                self.bn_names.append(name)
            if isinstance(W, nn.Linear):
                self.fc_names.append(name)

    def _encode(self, name):
        """
        Examples:
        conv1.weight -> conv           1                weight
                        conv1-> prefix   weight->postfix
                        conv->layer_type  1-> layer_id + 1  weight-> postfix
        Use buffer for efficient look up
        """
        prefix, postfix = name.split('.')
        dot_position = prefix.find('.')
        layer_id = ''
        for s in prefix:
            if s.isdigit():
                layer_id += s
        id_length = len(layer_id)
        layer_type = prefix[:-id_length]
        layer_id = int(layer_id) - 1
        if layer_type == 'conv' and len(self.conv_names) != 0:
            self.name_encoder[name] = self.conv_names[layer_id]
        elif layer_type == 'fc' and len(self.fc_names) != 0:
            self.name_encoder[name] = self.fc_names[layer_id]
        elif layer_type == 'bn' and len(self.bn_names) != 0:
            self.name_encoder[name] = self.bn_names[layer_id]
