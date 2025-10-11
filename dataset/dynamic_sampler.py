'''
# author: Yuchen Hui
# email: huiyc@stu.xidian.edu.cn
# date: 2025-10-11
# description: 动态采样器
'''

import torch
from torch.utils.data import Sampler
import numpy as np
import random

class DynamicRatioSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_ratio = 0.5
        self.fixed_ratio = 0.875 
        labels = dataset.data_dict['label']
        self.all_real_indices = [i for i, label in enumerate(labels) if label == 0]
        self.all_fake_indices = [i for i, label in enumerate(labels) if label == 1]
        

        self.real_indices = self.all_real_indices.copy()
        self.fake_indices = self.all_fake_indices.copy()
        random.shuffle(self.real_indices)
        random.shuffle(self.fake_indices)
        
        # if the real samples are exhausted
        self.real_exhausted = False
        # whether it is the first batch
        self.is_first_batch = True
        
        # Calculate the total number of batches (dynamic update)
        self.update_num_batches()

        self.reset_state()

        # Add loss tracking
        self.last_fake_loss = None
        self.last_real_loss = None
        self.epsilon = 1e-8

    def reset_state(self):
        self.real_indices = self.all_real_indices.copy()
        self.fake_indices = self.all_fake_indices.copy()
        random.shuffle(self.real_indices)
        random.shuffle(self.fake_indices)
        
        self.real_exhausted = False
        self.is_first_batch = True
        self.current_ratio = 0.5
        self.last_fake_loss = None
        self.last_real_loss = None
        self.update_num_batches()

    def update_num_batches(self):
        """Dynamic update batch number
        The characteristics of three stages:
        First stage:
            Only 1 batch
            Fixed 0.5:0.5 ratio (16:16)
        Second stage (dynamic stage):
            Ratio is dynamically adjusted between 0.3 and 0.7
            Batch number is determined by the number of real and fake samples
            Depends on which sample runs out under the current ratio
        Third stage (fixed stage):
            Fixed 1:7 ratio (4:28)
            Batch number is determined by the number of remaining fake samples
        """
        # If there are no fake samples, the mode degenerates to only real
        if len(self.all_fake_indices) == 0:
            self.num_batches = len(self.real_indices) // self.batch_size
            return
        if self.real_exhausted:
            n_real = int(self.batch_size * (1 - self.fixed_ratio))      # n_real = 4
            remaining_fake = len(self.fake_indices)  
            self.num_batches = remaining_fake // (self.batch_size - n_real)
        else:
            n_fake = int(self.batch_size * self.current_ratio)
            n_real = self.batch_size - n_fake
            remaining_real = len(self.real_indices)
            remaining_fake = len(self.fake_indices)
            batches_by_real = remaining_real // n_real if n_real > 0 else float('inf')
            batches_by_fake = remaining_fake // n_fake if n_fake > 0 else float('inf')
            self.num_batches = min(batches_by_real, batches_by_fake)
    
    def __iter__(self):
        self.reset_state()
        real_reuse_count = 0
        
        if len(self.fake_indices) == 0:
            while len(self.real_indices) >= self.batch_size:
                current_real_indices = self.real_indices[:self.batch_size]
                self.real_indices = self.real_indices[self.batch_size:]
                batch_indices = current_real_indices
                random.shuffle(batch_indices)
                yield batch_indices
            return
        
        while len(self.fake_indices) > 0:
            if self.is_first_batch:
                ratio = 0.5
                self.is_first_batch = False
            elif self.real_exhausted:
                ratio = self.fixed_ratio
            else:
                ratio = self.current_ratio
            
            n_fake = int(self.batch_size * ratio)
            n_real = self.batch_size - n_fake
            min_real_per_batch = 4
            n_real = max(min_real_per_batch, n_real)
            n_fake = self.batch_size - n_real
            
            if len(self.real_indices) < n_real:
                real_reuse_count += 1
                self.real_indices = self.all_real_indices.copy()
                random.shuffle(self.real_indices)
                if not self.real_exhausted:
                    self.real_exhausted = True
                    ratio = self.fixed_ratio
                    n_real = int(self.batch_size * (1 - self.fixed_ratio))
                    n_fake = self.batch_size - n_real
            

            if len(self.fake_indices) < n_fake:
                break

            current_real_indices = self.real_indices[:n_real]
            current_fake_indices = self.fake_indices[:n_fake]
            
            self.real_indices = self.real_indices[n_real:]
            self.fake_indices = self.fake_indices[n_fake:]

            self.update_num_batches()

            batch_indices = current_real_indices + current_fake_indices
            random.shuffle(batch_indices)
            stage = "First stage" if self.is_first_batch else ("Fixed stage" if self.real_exhausted else "Dynamic stage")
            
            yield batch_indices
    
    def __len__(self):
        return self.num_batches
        
    def update_ratio(self, fake_loss, real_loss):
        """
        Args:
            fake_loss (float): The loss of the fake images in the current batch (loss_ce_fake)
            real_loss (float): The loss of the real images in the current batch (loss_ce_real)
        """
        if not self.is_first_batch and not self.real_exhausted:
            if fake_loss > 0 or real_loss > 0:
                loss_ratio = fake_loss / (real_loss + self.epsilon) if real_loss > 0 else 10.0
                sigmoid_value = 1 / (1 + np.exp(-0.5 * (loss_ratio - 1)))
                new_ratio = 0.3 + sigmoid_value * 0.4 
                self.current_ratio = min(max(new_ratio, 0.3), 0.7)
            else:
                self.current_ratio = 0.5

            self.last_fake_loss = fake_loss
            self.last_real_loss = real_loss
            self.update_num_batches()