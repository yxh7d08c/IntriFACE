'''
# author: Yuchen Hui
# email: huiyc@stu.xidian.edu.cn
# date: 2025-10-11
# description: 训练检测器
'''

import os
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter


parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='./config/detector/intriface.yaml',
                    help='path to detector YAML file')  
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True) 
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True) 
parser.add_argument("--ddp", action='store_true', default=False)                           
parser.add_argument('--local_rank', type=int, default=-1)                                 
parser.add_argument('--task_target', type=str, default="", help='training IntriFace')
parser.add_argument('--resume_checkpoint', type=str, default='')
parser.add_argument('--resume_mode', type=str, choices=['continue','replay'], default='continue')
args = parser.parse_args()
if 'LOCAL_RANK' in os.environ and args.local_rank == -1:
    args.local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(args.local_rank if args.local_rank is not None else 0)


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_training_data(config):
    train_set = IntriFaceDataset(config, mode='train')
    
    if config['ddp']:
        sampler = DistributedSampler(train_set)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_dataloader']['batch_size'],
            num_workers=config['train_dataloader']['num_workers'],
            collate_fn=train_set.collate_fn,
            sampler=sampler
        )
    else:
        sampler = DynamicRatioSampler(
            dataset=train_set,
            batch_size=config['train_dataloader']['batch_size'] 
        )
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_sampler=sampler,
            num_workers=config['train_dataloader']['num_workers'],
            collate_fn=train_set.collate_fn,
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=2
        )
    
    return train_loader, sampler


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        test_config = config.copy()
        test_config['test_dataset'] = test_name
        
        test_set = IntriFaceDataset(
            config=test_config,
            mode='test'
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config['test_dataloader']['batch_size'], 
            shuffle=False,
            num_workers=config['test_dataloader']['num_workers'],
            collate_fn=test_set.collate_fn,
            drop_last=config['test_dataloader']['drop_last'],
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=2
        )

        return test_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs']/4),
        )
    elif config['lr_scheduler'] == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config['lr_milestones'],
            gamma=config['lr_gamma'],        
        )
        return scheduler
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def main():
    with open(args.detector_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    with open('./config/train_config.yaml', 'r', encoding='utf-8') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    config.update(config2)

    config['local_rank']=args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False

    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'

    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f"_{config['task_target']}" if config.get('task_target', None) is not None else ""
    logger_path =  os.path.join(
                config['log_dir'],
                config['model_name'] + task_str + '_' + timenow
            )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    config['ddp']= args.ddp
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))
    
    train_loader, sampler = prepare_training_data(config) 
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)  

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, 
                     train_loader=train_loader, sampler=sampler, time_now=timenow)

    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            ckpt = torch.load(args.resume_checkpoint, map_location='cpu')
            epoch_from_ckpt = ckpt.get('epoch', None)
            try:
                model_instance = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                state_dict = ckpt.get('model_state_dict', None)
                if state_dict is not None:
                    model_instance.load_state_dict(state_dict)
            except Exception as e:
                raise
            try:
                opt_state = ckpt.get('optimizer_state_dict', None)
                if opt_state is not None:
                    trainer.optimizer.load_state_dict(opt_state)
                sch_state = ckpt.get('scheduler_state_dict', None)
                if sch_state is not None and trainer.scheduler is not None:
                    trainer.scheduler.load_state_dict(sch_state)
            except Exception as e:
                logger.warning(f"Failed to restore the optimizer/scheduler state: {e}")
            if epoch_from_ckpt is not None:
                if args.resume_mode == 'continue':
                    config['start_epoch'] = epoch_from_ckpt + 1
                else:
                    config['start_epoch'] = epoch_from_ckpt

    # start training
    count = 0
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        if hasattr(trainer.model, 'set_epoch'):
            trainer.model.set_epoch(epoch - config['start_epoch'])
        trainer.model.epoch = epoch
        best_metric, count1 = trainer.train_epoch(
            epoch=epoch,
            test_data_loaders=test_data_loaders,
            count=count,
        )
        count = count1
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
    logger.info("Stop Training on best Testing metric {}".format(parse_metric_for_print(best_metric))) 
    # update
    if 'svdd' in config['model_name']:
        model.update_R(epoch)
    if scheduler is not None:
        scheduler.step()
    trainer.plot_losses(config['nEpochs'])
    for writer in trainer.writers.values():
        writer.close()
    if config['ddp']:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()