import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics
import matplotlib.pyplot as plt
from utils.memory_manager import MemoryManager

FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(self, config, model, optimizer, scheduler, logger, metric_scoring, 
                 train_loader=None, sampler=None, time_now=None):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.train_loader = train_loader
        self.sampler = sampler
        self.time_now = time_now
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.current_writers = set()
        
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()  # move model to GPU
        self.is_master = (not self.config['ddp']) or (dist.is_available() and (not dist.is_initialized() or dist.get_rank() == 0))

        # get current time
        self.timenow = time_now
        # create directory path
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        if self.is_master:
            os.makedirs(self.log_dir, exist_ok=True)

        self.loss_history = {
            'epoch': [],
            'total': [],
            'loss_bce': [],  
            'loss_ME': [],
            'loss_IR': [], 
            'loss_IGR': [] 
        }
        
        self.plot_dir = os.path.join(self.log_dir, 'loss_plots')
        if self.is_master:
            os.makedirs(self.plot_dir, exist_ok=True)
        self.epoch_save_dir = self.config.get('checkpoint_save_dir', os.path.join(self.log_dir, 'epoch_checkpoints'))
        if self.is_master:
            os.makedirs(self.epoch_save_dir, exist_ok=True)

        self.memory_manager = MemoryManager(logger)

    def _model_inst(self):
        return self.model.module if type(self.model) is DDP else self.model

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        self.current_writers.add(writer_key)
        if not self.is_master:
            class _Dummy:
                def add_scalar(self, *args, **kwargs):
                    pass
                def close(self):
                    pass
            return _Dummy()
        if writer_key not in self.writers:
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(self.model, device_ids=[self.config['local_rank']],find_unused_parameters=True, output_device=self.config['local_rank'])
            #self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))
        

    def save_ckpt(self, phase, dataset_key, ckpt_info=None):
        if not self.is_master:
            return
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "ckpt_best.pth")
        if self.config['ddp']:
            torch.save(self._model_inst().state_dict(), model_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({
                    'R': self.model.R,


                    'c': self.model.c,
                    'state_dict': self.model.state_dict(),
                }, model_path)
            else:
                torch.save(self.model.state_dict(), model_path)

        train_state_path = os.path.join(save_dir, "train_state_best.pth")
        save_info = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }
        if isinstance(ckpt_info, str):
            save_info['epoch_info'] = ckpt_info
        elif isinstance(ckpt_info, dict):
            save_info['epoch'] = ckpt_info.get('epoch')
        
        torch.save(save_info, train_state_path)

        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Training state saved to {train_state_path}")
        if ckpt_info:
            self.logger.info(f"Current checkpoint info: {ckpt_info}")


    def save_swa_ckpt(self):
        if not self.is_master:
            return
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, fea, dataset_key):
        if not self.is_master:
            return
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        if not self.is_master:
            return
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        if not self.is_master:
            return
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def clean_memory(self):
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()
        self.current_writers.clear()
        if hasattr(self, 'train_recorder_loss'):
            for recorder in self.train_recorder_loss.values():
                recorder.clear()
        if hasattr(self, 'train_recorder_metric'):
            for recorder in self.train_recorder_metric.values():
                recorder.clear()
        if hasattr(self, 'step_cnt'):
            del self.step_cnt
        if hasattr(self, 'last_test_step'):
            del self.last_test_step
        if hasattr(self, 'current_batch'):
            del self.current_batch
        self.memory_manager.clean()

    def train_step(self,data_dict):
        self.optimizer.zero_grad()
        base_model = self._model_inst()
        if hasattr(base_model, 'explicit_extractor') and base_model.explicit_extractor is not None:
            base_model.explicit_extractor.eval()

        if self.config['optimizer']['type']=='sam':
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['total'].backward(retain_graph=(i==0))
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:
            try:
                predictions = self.model(data_dict)
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)
                if hasattr(self, 'sampler') and self.sampler is not None:
                    fake_loss = losses['loss_ce_fake']
                    real_loss = losses['loss_ce_real']
                    if hasattr(self.sampler, 'update_ratio'):
                        self.sampler.update_ratio(fake_loss.item(), real_loss.item())
                losses['total'].backward()
                self.optimizer.step()
                return losses, predictions
            except Exception as e:
                import traceback
                self.logger.error(traceback.format_exc())
                raise

    def train_epoch(
        self,
        epoch,
        test_data_loaders=None,
        count=0,
        ):
        self.setTrain()
        base_model = self._model_inst()
        if hasattr(base_model, 'explicit_extractor') and base_model.explicit_extractor is not None:
            for param in base_model.explicit_extractor.parameters():
                param.requires_grad = False
        if hasattr(self, 'sampler') and self.sampler is not None:
            if hasattr(self.sampler, 'set_epoch'):
                self.sampler.set_epoch(epoch)
            elif hasattr(self.sampler, 'reset_state'):
                self.sampler.reset_state()

        self.memory_manager.check_memory(f"Epoch {epoch} start")

        times_per_epoch = 4
        total_steps = len(self.train_loader)
        test_step = total_steps // times_per_epoch
        step_cnt = 0
        last_test_step = 0 
        test_best_metric = None
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)


        epoch_losses = {
            'total': 0.0, 
            'loss_bce': 0.0,
            'loss_ME': 0.0,  
            'loss_IR': 0.0, 
            'loss_IGR': 0.0 
        }
        processed_batches = 0
        
        def run_test(count):
            if test_data_loaders is not None:
                if not self.config['ddp'] or (self.config['ddp'] and dist.get_rank() == 0):
                    self.logger.info("===> Test start!")
                    return self.test_epoch(epoch, iteration, test_data_loaders, count)
            return None

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}", 
                   ncols=100, leave=True, position=0, dynamic_ncols=True)
        for iteration, data_dict in pbar:
            try:
                for key in data_dict.keys():
                    if data_dict[key]!=None and key!='name':
                        data_dict[key]=data_dict[key].cuda()
                losses,predictions=self.train_step(data_dict)
                if iteration == 10 and last_test_step == 0:
                    test_best_metric = run_test(count)
                    last_test_step = step_cnt

                # compute training metric for each batch data
                if type(self.model) is DDP:
                    batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
                else:
                    batch_metrics = self.model.get_train_metrics(data_dict, predictions)

                # store data by recorder
                ## store metric
                for name, value in batch_metrics.items():
                    train_recorder_metric[name].update(value)
                ## store loss
                for name, value in losses.items():
                    train_recorder_loss[name].update(value)
                if iteration % 1000 == 0:
                    if self.config['SWA'] and (epoch>self.config['swa_start'] or self.config['dry_run']):
                        self.scheduler.step()
                    # info for loss
                    loss_str = f"Iter: {count}    "
                    for k, v in train_recorder_loss.items():
                        v_avg = v.average()
                        if v_avg == None:
                            loss_str += f"training-loss, {k}: not calculated"
                            continue
                        loss_str += f"training-loss, {k}: {v_avg}    "
                        # tensorboard-1. loss
                        writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                        writer.add_scalar(f'train_loss/{k}', v_avg, global_step=count)
                    self.logger.info(loss_str)
                    # info for metric
                    metric_str = f"Iter: {count}    "
                    for k, v in train_recorder_metric.items():
                        v_avg = v.average()
                        if v_avg == None:
                            metric_str += f"training-metric, {k}: not calculated    "
                            continue
                        metric_str += f"training-metric, {k}: {v_avg}    "
                        # tensorboard-2. metric
                        writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                        writer.add_scalar(f'train_metric/{k}', v_avg, global_step=count)
                    self.logger.info(metric_str)



                    # clear recorder.
                    # Note we only consider the current 300 samples for computing batch-level loss/metric
                    for name, recorder in train_recorder_loss.items():  # clear loss recorder
                        recorder.clear()
                    for name, recorder in train_recorder_metric.items():  # clear metric recorder
                        recorder.clear()
                steps_since_last_test = step_cnt - last_test_step
                if steps_since_last_test >= test_step and iteration > 10:
                    test_best_metric = run_test(count)
                    last_test_step = step_cnt
                step_cnt += 1
                
                for key in epoch_losses.keys():
                    if key in losses:
                        epoch_losses[key] += losses[key].item()
                processed_batches += 1
                
                current_batches = len(self.train_loader)
                loss_str = f'Epoch {epoch} [{processed_batches}/{current_batches}]'
                
                detailed_loss_keys = [
                    'loss_pf_cross', 'loss_pf_inner', 'loss_ie_fake', 'loss_ie_real',
                    'loss_IR_real', 'loss_IR_fake',                                  
                    'loss_IGR',                                                      
                    'loss_ME', 'loss_IR', 'loss_bce'                                  
                ]
                
                for loss_key in detailed_loss_keys:
                    if loss_key in losses and isinstance(losses[loss_key], torch.Tensor):
                        short_key = loss_key.replace('loss_', '')
                        loss_str += f' {short_key}={losses[loss_key].item():.4f}'
                
                if 'total' in losses:
                    loss_str += f' total={losses["total"].item():.4f}'
                
                if hasattr(self.optimizer, 'param_groups'):
                    current_lr = self.optimizer.param_groups[0]['lr']
                    loss_str += f' lr={current_lr:.6f}'
                
                pbar.set_description(loss_str)

            finally:
                if 'predictions' in locals(): del predictions
                if 'data_dict' in locals(): del data_dict
                if 'losses' in locals(): del losses
                if iteration % 10 == 0:
                    torch.cuda.empty_cache()
            count += 1
        if processed_batches > 0:
            self.loss_history['epoch'].append(epoch)
            self.loss_history['total'].append(epoch_losses['total'] / processed_batches)
            self.loss_history['loss_bce'].append(epoch_losses['loss_bce'] / processed_batches)
            self.loss_history['loss_ME'].append(epoch_losses['loss_ME'] / processed_batches)
            self.loss_history['loss_IR'].append(epoch_losses['loss_IR'] / processed_batches)
            self.loss_history['loss_IGR'].append(epoch_losses['loss_IGR'] / processed_batches)
        
        self.save_epoch_checkpoint(epoch)
        self.clean_memory()
        
        return test_best_metric, count

    def get_respect_acc(self,prob,label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        zero_num = len(label) - np.count_nonzero(label)
        acc_fake = np.count_nonzero(judge[zero_num:]) / len(judge[zero_num:])
        acc_real = np.count_nonzero(judge[:zero_num]) / len(judge[:zero_num])
        return acc_real,acc_fake

    def test_one_dataset(self, data_loader):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists=[]
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader),total=len(data_loader)):
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only
            # move data to GPU elegantly
            for key in data_dict.keys():
                if data_dict[key]!=None:
                    data_dict[key]=data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists += list(predictions['feat'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)
                    
                # store data by recorder
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)

    def save_best(self,epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                    metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            self.save_metrics('test', metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                # tensorboard-1. loss
                writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)
                loss_str += f"testing-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k=='dataset_dict':
                continue
            metric_str += f"testing-metric, {k}: {v}    "
            # tensorboard-2. metric
            writer = self.get_writer('test', key, k)
            writer.add_scalar(f'test_metrics/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar(f'test_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'test_metrics/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)
    
    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
        # testing for all test data
        keys = test_data_loaders.keys()
        for key in keys:
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.test_one_dataset(test_data_loaders[key])
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset=get_test_metrics(y_pred=predictions_nps,y_true=label_nps,img_names=data_dict['image'])
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset)

        if len(keys)>0 and self.config.get('save_avg',False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    def save_epoch_checkpoint(self, epoch):
        if not self.is_master:
            return
        model_instance = self.model.module if type(self.model) is DDP else self.model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_instance.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }
        
        save_path = os.path.join(self.epoch_save_dir, f'epoch_{epoch}.pth')
        torch.save(checkpoint, save_path)

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions