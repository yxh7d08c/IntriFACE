'''
# author: Yuchen Hui
# email: huiyc@stu.xidian.edu.cn
# date: 2025-10-11
# description: 检测器训练
'''

import os
import datetime
import logging
import random

import numpy as np
import yaml
from sklearn import metrics
from typing import Union
from collections import defaultdict

from dataset.intriFace_dataset import IntriFaceDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from models.iib import IIB
from metrics.base_metrics_class import calculate_metrics_for_train
from models.encoder128 import Backbone128
from models.encoder128 import load_custom_state_dict
from detectors.base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE

logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)

def l2_norm(input, axis=1):
    norm = torch.norm(input, p=2, dim=axis, keepdim=True)
    output = torch.div(input, norm)
    return output


class IRModule(nn.Module):
    """
    Identity Refinement module, used to explicitly delete the part of Mixed Identity that is irrelevant to the target identity
    """
    def __init__(self, feature_dim=512):
        super(IRModule, self).__init__()
        # Define the nonlinear residual structure
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv_relu = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.ReLU()
        )
        self.final_add = nn.Identity()  # Used for residual connection
        
    def forward(self, mixed_identity, f_remove):
        """
        Args:
            mixed_identity: Mixed identity feature [batch_size, feature_dim]
            f_remove: Identity component to be removed [batch_size, feature_dim]
        Returns:
            refined_identity: Refined identity feature [batch_size, feature_dim]
        """
        # Ensure that the two input tensors are on the same device
        device = mixed_identity.device
        f_remove = f_remove.to(device)
        
        # Convert the feature to the channel-first format [batch_size, feature_dim, 1]
        mixed_identity_3d = mixed_identity.unsqueeze(-1)
        f_remove_3d = f_remove.unsqueeze(-1)
        
        # Ensure that the module is also on the correct device
        self.avgpool = self.avgpool.to(device)
        self.conv_relu = self.conv_relu.to(device)
        self.final_add = self.final_add.to(device)
        
        # Apply average pooling
        pooled = self.avgpool(f_remove_3d)  # [batch_size, feature_dim, 1]
        
        # Apply the first multiplication operation
        mult1 = mixed_identity_3d * pooled  # [batch_size, feature_dim, 1]
        mult2 = mult1 * pooled
        
        # Apply the second multiplication operation (multiply by -1)
        mult3 = mult2 * (-1)  # [batch_size, feature_dim, 1]
        
        # Apply the addition operation
        added = mixed_identity_3d + mult3  # [batch_size, feature_dim, 1]

        # Apply the convolution and ReLU
        conv_out = self.conv_relu(added)  # [batch_size, feature_dim, 1]

        refined = self.final_add(conv_out + 0.13 * mixed_identity_3d)  # [batch_size, feature_dim, 1]
        return refined.squeeze(-1)  # [batch_size, feature_dim]

    def compute_projection(self, source_identity, target_identity):
        """
        Calculate the projection of the source identity on the target identity direction
        Args:
            source_identity: Source identity feature [batch_size, feature_dim]
            target_identity: Target identity feature [batch_size, feature_dim]
        Returns:
            projection: Projection component [batch_size, feature_dim]
            f_remove: Identity component to be removed [batch_size, feature_dim]
        """
        source_norm = F.normalize(source_identity, p=2, dim=1)
        target_norm = F.normalize(target_identity, p=2, dim=1)
        dot_product = torch.sum(source_norm * target_norm, dim=1, keepdim=True)  # [batch_size, 1]
        projection = dot_product * target_norm  # [batch_size, feature_dim]
        f_remove = source_norm - projection  # [batch_size, feature_dim]
        return projection, f_remove

class IGRModule(nn.Module):
    """
    Identity Gain Refinement module, used to purify the residual identity
    """
    def __init__(self, feature_dim=512):
        super(IGRModule, self).__init__()
        # Global pooling layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature projection layer
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim//2, feature_dim)
        )
        
        # Correlation evaluation module - evaluate the correlation between y and yr
        self.correlation = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
        # Information entropy estimator - used to evaluate the information content of the feature
        self.entropy_estimator = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim//2, 3, padding=1),
            nn.InstanceNorm1d(feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim//2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, y, yr):
        """
        y: Weakened identity feature [batch_size, feature_dim]
        yr: Target explicit identity feature [batch_size, feature_dim] (need to retain the part related to it)
        """
        device = y.device
        yr = yr.to(device)
        self.global_pool = self.global_pool.to(device)
        self.projection = self.projection.to(device)
        self.correlation = self.correlation.to(device)
        self.entropy_estimator = self.entropy_estimator.to(device)
        
        batch_size, feature_dim = y.shape
        y_3d = y.unsqueeze(-1)  # [batch_size, feature_dim, 1]
        yr_3d = yr.unsqueeze(-1)  # [batch_size, feature_dim, 1]
        y_global = self.global_pool(y_3d).view(batch_size, -1)  # [batch_size, feature_dim]
        yr_global = self.global_pool(yr_3d).view(batch_size, -1)  # [batch_size, feature_dim]
        combined = torch.cat([y_global, yr_global], dim=1)  # [batch_size, feature_dim*2]
        relevance_weights = self.correlation(combined).view(batch_size, feature_dim, 1)  # [batch_size, feature_dim, 1]
        weighted_features = y_3d * relevance_weights  # [batch_size, feature_dim, 1]
        if weighted_features.size(2) == 1:
            weighted_features_expanded = weighted_features.repeat(1, 1, 2)  # [batch_size, feature_dim, 2]
            conv1 = self.entropy_estimator[0]
            instance_norm = self.entropy_estimator[1]
            relu = self.entropy_estimator[2]
            conv2 = self.entropy_estimator[3]
            sigmoid = self.entropy_estimator[4]
            x = conv1(weighted_features_expanded)  # [batch_size, feature_dim//2, 2]
            x = instance_norm(x)
            x = relu(x)
            x = x[:, :, 0:1]  # [batch_size, feature_dim//2, 1]
            x = conv2(x)  # [batch_size, 1, 1]
            entropy_weights = sigmoid(x)
        else:
            entropy_weights = self.entropy_estimator(weighted_features)  # [batch_size, 1, 1]

        preserved_features = y_3d * (relevance_weights * entropy_weights)  # [batch_size, feature_dim, 1]
        enhanced_features = preserved_features + self.projection(y_global).view(batch_size, feature_dim, 1) * entropy_weights
        return enhanced_features.squeeze(-1)  # [batch_size, feature_dim]

    def get_pured_identity(self, weaked_identity, target_index, env_id, em_kernel):
        """
        Use the IGR module to purify the residual identity
        
        Args:
            weaked_identity: Identity feature after the IR module [batch_size, feature_dim]
            target_index: Target identity ID [batch_size]
            env_id: Environment ID [batch_size]
            em_kernel: Explicit feature library
            
        Returns:
            pured_identity: Purified residual identity feature [batch_size, feature_dim]
        """
        batch_size = weaked_identity.size(0)
        device = weaked_identity.device
        pured_identity = torch.zeros_like(weaked_identity)
        target_index = target_index.to(device)
        env_id = env_id.to(device)
        
        for i in range(batch_size):
            current_target_idx = target_index[i].item()
            current_env_id = env_id[i].item()
            current_weaked = weaked_identity[i:i+1]
            target_identity = None
            
            target_sub_kernel = em_kernel.get(current_target_idx)
            if target_sub_kernel is not None and current_env_id < target_sub_kernel.size(1):
                target_identity = target_sub_kernel[:, current_env_id].unsqueeze(0).to(device)  # [1, feature_dim]
            if target_identity is not None:
                pured_identity[i:i+1] = self(current_weaked, target_identity)
            else:
                pured_identity[i:i+1] = current_weaked
        return pured_identity

@DETECTOR.register_module(module_name='intriface')
class IntriFace(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        em_kernel_path = config.get('explicit_feature_bank_path', None)
        if em_kernel_path and os.path.exists(em_kernel_path):
            checkpoint = torch.load(em_kernel_path)
            self.em_kernel = checkpoint['em_kernel']
            self.missing_ids = set(checkpoint['missing_ids'])
            logger.info(f"Loaded explicit kernel from {em_kernel_path}")
            logger.info(f"Missing IDs: {self.missing_ids}")
        else:
            raise ValueError("em_kernel_path not specified or file not found")

        self.im_kernel = {}
        self.feature_dim = config['embedding_size']
        for person_id, feature_tensor in self.em_kernel.items():
            self.im_kernel[person_id] = {}
            for env_id in range(feature_tensor.size(1)):
                self.im_kernel[person_id][env_id] = torch.zeros(1, self.feature_dim)
        
        self.current_epoch = 0
        self.should_update_im_kernel = False
    

        self.config = config
        self.backbone = self.build_backbone(config)


        netArc_checkpoint_path = self.config.get('explicit_extractor_pretrained', None)
        if netArc_checkpoint_path:
            netArc_checkpoint = torch.load(netArc_checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
            self.explicit_extractor = netArc_checkpoint
            self.explicit_extractor.cuda().eval()
            for param in self.explicit_extractor.parameters():
                param.requires_grad = False
        self.N = 10
        self.encoder = Backbone128(50, 0.6, 'ir_se').eval()
        encoder_path = config.get('encoder_path', None)
        original_state_dict = torch.load(encoder_path, map_location='cpu')
        load_custom_state_dict(self.encoder, original_state_dict)
        for param in self.encoder.parameters():
            param.requires_grad = False
        with torch.no_grad():
            _ = self.encoder(torch.rand(1, 3, 128, 128), cache_feats=True)
            _readout_feats = self.encoder.features[:(self.N + 1)]
            in_c = sum(map(lambda f: f.shape[-3], _readout_feats))
            out_c_list = [_readout_feats[i].shape[-3] for i in range(self.N)]
            self.encoder.features = [] 

        self.iib = IIB(in_c, out_c_list, 'cuda', smooth=True, kernel_size=1)
        self.iib = self.iib.eval()
        for param in self.iib.parameters():
            param.requires_grad = False
        self.iib = IIB(in_c, out_c_list, 'cuda', smooth=True, kernel_size=1)
        iib_checkpoint_path = self.config.get('iib_checkpoint_path', None)
        state = torch.load(iib_checkpoint_path, map_location='cpu')
        self.iib.load_state_dict(state, strict=True)
        logger.info(f"Loaded IIB weights from {iib_checkpoint_path}")
        self.iib = self.iib.eval()
        for param in self.iib.parameters():
            param.requires_grad = False
        self.ir_module = IRModule(feature_dim=config['embedding_size'])
        self.igr_module = IGRModule(feature_dim=config['embedding_size'])
        self.alpha_1 = config.get('alpha_1', 1)
        self.kappa = config.get('kappa', 0.5)
        self.beta_1 = config.get('beta_1', 1)
        self.beta_2 = config.get('beta_2', 1)
        self.beta_3 = config.get('beta_3', 1)
        self.k = config.get('k', 1)
        self.Lambda = config.get('lambda', 1)
        
    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        if config['pretrained'] != 'None':
            # if donot load the pretrained weights, fail to get good results
            state_dict = torch.load(config['pretrained'])
            for name, weights in state_dict.items():
                if 'pointwise' in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
            backbone.load_state_dict(state_dict, False)
            logger.info('Load pretrained model successfully!')
        else:
            logger.info('No pretrained model.')
        return backbone
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.should_update_im_kernel = (epoch >= 1)
        logger.info(f"Set epoch to {epoch}, should_update_im_kernel: {self.should_update_im_kernel}")
        
    def update_im_kernel(self, mixed_identity, target_index, env_id, label):
        """
        Update the residual identity bank
        
        Args:
            mixed_identity: Mixed identity feature [batch_size, feature_dim]
            target_index: Target identity ID [batch_size]
            env_id: Environment ID [batch_size]
            label: Label [batch_size] (0:real, 1:fake)
        """
        if not self.should_update_im_kernel:
            return
        
        real_id = (label == 0)
        if not real_id.any():
            return
        
        # Get the mixed identity, target ID, and environment ID of the real images
        try:
            logger.debug(f"real_id: {real_id.sum().item()}/{label.size(0)}, mixed_identity.shape: {mixed_identity.shape}")
            
            real_mixed_identity = mixed_identity[real_id]  # [num_real, feature_dim]
            real_target_index = target_index[real_id]  # [num_real]
            real_env_id = env_id[real_id]  # [num_real]
            for i in range(real_mixed_identity.size(0)):
                person_id = real_target_index[i].item()
                current_env_id = real_env_id[i].item()
                if person_id in self.im_kernel:
                    if current_env_id not in self.im_kernel[person_id]:
                        self.im_kernel[person_id][current_env_id] = torch.zeros(1, self.feature_dim, device=mixed_identity.device)
                    self.im_kernel[person_id][current_env_id] = real_mixed_identity[i:i+1].detach() 

            if random.random() < 0.01: 
                total_persons = len(self.im_kernel)
                total_envs = sum(len(envs) for envs in self.im_kernel.values())
                logger.info(f"Updated residual identity bank, total persons: {total_persons}, total environments: {total_envs}")
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())

    def get_im_feature(self, person_id, env_id):
        if person_id in self.im_kernel and env_id in self.im_kernel[person_id]:
            return self.im_kernel[person_id][env_id]
        return None

    def get_em_feature(self, data_dict: dict) -> torch.tensor:
        device = data_dict['image'].device
        
        with torch.no_grad():
            X_id = self.encoder(
                F.interpolate(data_dict['image'], size=[128, 128],
                                mode='bilinear', align_corners=True),
                cache_feats=True
            )
            # 01 Get Inter-features After One Feed-Forward:
            param_dict = []
            for i in range(self.N + 1):
                weights_path = os.path.join(self.config.get('weights128_celeb_path', './models/weights128_celeb'), f'readout_layer{i}.pth')
                state = torch.load(weights_path, map_location=device)
                n_samples = state['n_samples'].float()
                std = torch.sqrt(state['s'] / (n_samples - 1)).to(device)
                neuron_nonzero = state['neuron_nonzero'].float()
                active_neurons = (neuron_nonzero / n_samples) > 0.01
                param_dict.append([state['m'].to(device), std, active_neurons.to(device)])
            # batch size is 2 * B, [:B] for Xs and [B:] for Xt
            min_std = torch.tensor(0.01, device=device)
            readout_feats = [(self.encoder.features[i] - param_dict[i][0]) / torch.max(param_dict[i][1], min_std)
                                for i in range(self.N + 1)]

            # 02 information restriction:
            X_id_restrict = torch.zeros_like(X_id).to(device)
            for i in range(self.N):
                R = self.encoder.features[i] 
                Z, _, _ = getattr(self.iib, f'iba_{i}')(
                    R, readout_feats,
                    m_r=param_dict[i][0], std_r=param_dict[i][1],
                    active_neurons=param_dict[i][2],
                )
                X_id_restrict += self.encoder.restrict_forward(Z, i)
            X_id_restrict /= float(self.N)
            self.encoder.features = []
        return X_id_restrict

    def get_weaked_identity(self, mixed_identity, source_index, target_index, env_id, label):
        batch_size = mixed_identity.size(0)
        feature_dim = mixed_identity.size(1)
        device = mixed_identity.device
        weaked_identity = torch.zeros_like(mixed_identity)
        source_identity = torch.zeros_like(mixed_identity)  # [batch_size, feature_dim]
        target_identity = torch.zeros_like(mixed_identity)  # [batch_size, feature_dim]
        source_index = source_index.to(device)
        target_index = target_index.to(device)
        env_id = env_id.to(device)
        label = label.to(device)
        
        for i in range(batch_size):
            current_source_idx = source_index[i].item()
            current_target_idx = target_index[i].item()
            current_env_id = env_id[i].item()
            current_mixed = mixed_identity[i:i+1]
            current_source_identity = None
            current_target_identity = None

            source_sub_kernel = self.em_kernel.get(current_source_idx)
            if source_sub_kernel is not None and source_sub_kernel.size(1) > 0:
                valid_features = []
                for env in range(source_sub_kernel.size(1)):
                    valid_features.append(source_sub_kernel[:, env])
                if valid_features:
                    current_source_identity = torch.stack(valid_features, dim=0).mean(dim=0).unsqueeze(0).to(device)  # [1, feature_dim]
                    source_identity[i:i+1] = current_source_identity

            target_sub_kernel = self.em_kernel.get(current_target_idx)
            if target_sub_kernel is not None and current_env_id < target_sub_kernel.size(1):
                current_target_identity = target_sub_kernel[:, current_env_id].unsqueeze(0).to(device)  # [1, feature_dim]
                target_identity[i:i+1] = current_target_identity

            if current_source_identity is not None and current_target_identity is not None:
                current_source_identity = F.normalize(current_source_identity, p=2, dim=1)
                current_target_identity = F.normalize(current_target_identity, p=2, dim=1)
                dot_product = torch.sum(current_source_identity * current_target_identity, dim=1, keepdim=True)  # [1, 1]
                projection = dot_product * current_target_identity  # [1, feature_dim]
                f_remove = current_source_identity - projection  # [1, feature_dim]
                current_weaked_identity = self.ir_module(current_mixed, f_remove)
                weaked_identity[i:i+1] = current_weaked_identity
            else:
                weaked_identity[i:i+1] = current_mixed
            
        return weaked_identity, source_identity, target_identity

    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image']) #32,3,256,256

    def classifier(self, features: torch.tensor,id_f=None) -> torch.tensor:
        return self.backbone.classifier(features,id_f)

    def get_train_loss(self, data_dict: dict, pred_dict: dict) -> dict:
        # region 0: datadict refers to the attributes of data itself
        label = data_dict['label'].cuda()
        source_index = data_dict['source_index'].cuda()
        target_index = data_dict['target_index'].cuda() 
        envID = data_dict['env_id'].cuda() 
        flag = data_dict['flag'].cuda() 
        label = label * (1 - flag * label)
        real_id = (label == 0)
        fake_id = (label == 1)
        # endregion
        
        # region 1: after forward, get the results: specific prediction results and various identities
        fake_prob = pred_dict['prob'].cuda()
        explicit_Identity = pred_dict['id_feat'].cuda()    
        pure_Identity = pred_dict['embed'].cuda()           
        mixed_Identity = pred_dict['mixed_identity'].cuda() 
        weaked_Identity = pred_dict['weaked_identity'].cuda() 
        source_Identity = pred_dict['source_identity'].cuda() 
        target_Identity = pred_dict['target_identity'].cuda() 
        em_idt = l2_norm(explicit_Identity)
        pim_idt = l2_norm(pure_Identity)
        mixed_idt = l2_norm(mixed_Identity)
        weaked_idt = l2_norm(weaked_Identity)
        source_idt = l2_norm(source_Identity)
        target_idt = l2_norm(target_Identity)
        loss = 0
        # endregion

        # region 2: first part loss: classification head loss
        loss_ce_real = torch.tensor(0.0).cuda()
        loss_ce_fake = torch.tensor(0.0).cuda()

        if real_id.any():
            real_fake_prob = fake_prob[real_id]
            real_label = torch.zeros_like(real_fake_prob, device=fake_prob.device)
            loss_ce_real = F.binary_cross_entropy_with_logits(
                real_fake_prob,
                real_label,
                reduction='mean'
            )
        if fake_id.any():
            fake_fake_prob = fake_prob[fake_id]
            fake_label = torch.ones_like(fake_fake_prob, device=fake_prob.device)
            loss_ce_fake = F.binary_cross_entropy_with_logits(
                fake_fake_prob,
                fake_label,
                reduction='mean'
            )
        
        loss_sim = torch.tensor(0.0).cuda()
        if real_id.any():
            loss_sim -= (pim_idt[real_id] * em_idt[real_id]).sum(dim=1).mean()
        if fake_id.any():
            loss_sim += (pim_idt[fake_id] * em_idt[fake_id]).sum(dim=1).mean()
        loss_ce = loss_ce_real + loss_ce_fake + 2 * loss_sim
        # endregion

        # region 3: second part loss: ME module loss
        ## perceptual filtering loss: consistency constraint of explicit and residual identity similarity across environments
        loss_pf_cross = torch.tensor(0.0).cuda()
        
        # only calculate this loss after the residual identity bank starts updating (second epoch starts)
        if self.should_update_im_kernel and real_id.any():
            real_mixed_id = mixed_idt[real_id]  # [num_real, feature_dim]
            real_person_ids = target_index[real_id]
            real_env_ids = envID[real_id]  # [num_real]

            cross_env_losses = []
            for i in range(real_person_ids.size(0)):
                person_id = real_person_ids[i].item()
                current_env_id = real_env_ids[i].item()
                current_mixed_id = real_mixed_id[i:i+1]  # [1, feature_dim]
                

                other_env_explicit_ids = []
                other_env_mixed_ids = []

                if person_id in self.em_kernel:
                    em_feature_tensor = self.em_kernel[person_id]

                    if current_env_id < em_feature_tensor.size(1):
                        current_explicit_id = em_feature_tensor[:, current_env_id].unsqueeze(0).to(current_mixed_id.device)  # [1, feature_dim]
                    else:
                        continue

                    for env_id in range(em_feature_tensor.size(1)):
                        if env_id == current_env_id:
                            continue

                        other_explicit_id = em_feature_tensor[:, env_id].unsqueeze(0)  # [1, feature_dim]
                        other_explicit_id = other_explicit_id.to(current_mixed_id.device)
    
                        if person_id in self.im_kernel and env_id in self.im_kernel[person_id]:
                            other_mixed_id = self.im_kernel[person_id][env_id]  # [1, feature_dim]
                   
                            other_mixed_id = other_mixed_id.to(current_mixed_id.device)
                            
                            other_env_explicit_ids.append(other_explicit_id)
                            other_env_mixed_ids.append(other_mixed_id)
                else:
                    continue

                if other_env_explicit_ids and other_env_mixed_ids:
                    for j in range(len(other_env_explicit_ids)):
                        current_explicit_id_norm = F.normalize(current_explicit_id, p=2, dim=1)
                        other_explicit_id_norm = F.normalize(other_env_explicit_ids[j], p=2, dim=1)
                        current_mixed_id_norm = F.normalize(current_mixed_id, p=2, dim=1)
                        other_mixed_id_norm = F.normalize(other_env_mixed_ids[j], p=2, dim=1)
                        with torch.no_grad():
                            explicit_sim = F.cosine_similarity(current_explicit_id_norm, other_explicit_id_norm, dim=1)
                        mixed_sim = F.cosine_similarity(current_mixed_id_norm, other_mixed_id_norm.detach(), dim=1)
                        C1 = 0.01
                        sim_diff = torch.abs(((mixed_sim + C1)/(explicit_sim + C1)) - 1)
                        cross_env_losses.append(sim_diff)
            if cross_env_losses:
                loss_pf_cross = torch.mean(torch.cat(cross_env_losses))

        ## calculate loss_pf_inner: KL divergence of von Mises-Fisher distribution
        # only calculate the loss of real samples
        if real_id.any():
            real_explicit_id = em_idt[real_id]  # [num_real, feature_dim]
            real_mixed_id = mixed_idt[real_id]  # [num_real, feature_dim]
            dot_product = torch.sum(real_explicit_id * real_mixed_id, dim=1)  # [num_real]
            kl_divergence = self.kappa * (1 - dot_product)  # [num_real]
            loss_pf_inner = kl_divergence.mean()
        else:
            loss_pf_inner = torch.tensor(0.0).cuda()
        
        loss_pf = loss_pf_cross + 3 * loss_pf_inner

        ## identity extraction loss: truncated exponential distribution
        if fake_id.any():
            # get the mixed identity, source identity, and target identity of the fake samples
            fake_mixed_id = mixed_idt[fake_id]  # [num_fake, feature_dim]
            fake_source_id = source_idt[fake_id].detach()  # [num_fake, feature_dim]
            fake_target_id = target_idt[fake_id].detach()  # [num_fake, feature_dim]
            with torch.no_grad():
                cos_st = torch.sum(fake_source_id * fake_target_id, dim=1)  # [num_fake]
                theta_max = torch.acos(torch.clamp(cos_st, -0.99999, 0.99999))  # [num_fake]
                u = torch.rand_like(theta_max)  # [num_fake]
                theta_half = theta_max / 2
                theta_deviation = -torch.log(1 - u * (1 - torch.exp(-self.Lambda * theta_half))) / self.Lambda + theta_half
                theta_deviation = torch.clamp(theta_deviation, theta_half, theta_max)

            cos_fs = torch.sum(fake_mixed_id * fake_source_id, dim=1)  # [num_fake]
            cos_ft = torch.sum(fake_mixed_id * fake_target_id, dim=1)  # [num_fake]
            cos_deviation = torch.cos(theta_deviation)  # [num_fake]
            min_cos = torch.minimum(cos_fs, cos_ft)
            loss_ie_fake = torch.clamp(cos_deviation - min_cos, min=0).mean()
        else:
            loss_ie_fake = torch.tensor(0.0).cuda()

        with torch.no_grad():
            if fake_id.any():
                theta_margin_value = (theta_max.mean() / 2 - torch.acos(torch.clamp(cos_fs.mean(), -0.99999, 0.99999))).item()
            else:
                theta_margin_value = 0.0
            
        if real_id.any():
            with torch.no_grad():
                emd_idt_real_detached = em_idt[real_id].detach()
            cos_real = torch.sum(mixed_idt[real_id] * emd_idt_real_detached, dim=1)  # [num_real]
            theta_real = torch.acos(torch.clamp(cos_real, -0.99999, 0.99999))  # [num_real]
            theta_margin_tensor = torch.full_like(theta_real, theta_margin_value)
            valid_angles = theta_real - theta_margin_tensor
            valid_mask = valid_angles > 0
            
            if valid_mask.any():
                loss_ie_real = torch.log(valid_angles[valid_mask]).mean()
            else:
                loss_ie_real = torch.tensor(0.0).cuda()
        else:
            loss_ie_real = torch.tensor(0.0).cuda()
        
        loss_ie = 1.7 * loss_ie_fake + loss_ie_real

        ## ME module total loss
        loss_ME = loss_pf + self.alpha_1 * loss_ie
        # endregion

        # region 4: third part loss: IR module loss  
        # for the IR module loss of real images, it is to constrain the deletion before and after as close as possible
        if real_id.any():
            loss_IR_real = F.cosine_similarity(target_idt[real_id], weaked_idt[real_id], dim=1).mean()
        else:
            loss_IR_real = torch.tensor(0.0).cuda()

        if fake_id.any():
            with torch.no_grad():
                target_idt_fake_detached = target_idt[fake_id].detach()
                mixed_idt_fake_detached = mixed_idt[fake_id].detach()
            mixed_sim = F.cosine_similarity(mixed_idt_fake_detached, target_idt_fake_detached, dim=1).mean()
            weaked_sim = F.cosine_similarity(weaked_idt[fake_id], target_idt_fake_detached, dim=1).mean()
            loss_IR_fake = mixed_sim - weaked_sim
        else:
            loss_IR_fake = torch.tensor(0.0).cuda()

        loss_IR = 4 * loss_IR_fake - loss_IR_real
        # endregion

        # region 5: fourth part loss: IGR module loss
        # use detach() to prevent gradient propagation to target_idt and weaked_idt
        with torch.no_grad():
            target_idt_detached = target_idt.detach()
            weaked_idt_detached = weaked_idt.detach()
            target_norm_squared = torch.sum(target_idt_detached * target_idt_detached, dim=1, keepdim=True)
            weaked_projection = (weaked_idt_detached * target_idt_detached).sum(1, keepdim=True) * target_idt_detached / (target_norm_squared + 1e-8)
            weaked_proj_norm = torch.norm(weaked_projection, dim=1)
        pim_projection = (pim_idt * target_idt_detached).sum(1, keepdim=True) * target_idt_detached / (target_norm_squared + 1e-8)
        pim_proj_norm = torch.norm(pim_projection, dim=1)
        loss_IGR = -1 * torch.log(1 + torch.exp(pim_proj_norm - weaked_proj_norm)).mean()
        # endregion

        # region 6:
        loss = loss_ce + self.beta_1 * loss_ME + self.beta_2 * loss_IR + self.beta_3 * loss_IGR
        loss_dict = {
            'total': loss,
            'loss_ce_real': loss_ce_real,
            'loss_ce_fake': loss_ce_fake,
            'loss_bce': loss_ce,
            'loss_ME': loss_ME, 
            'loss_IR': loss_IR, 
            'loss_IGR': loss_IGR,
            'loss_pf_cross': loss_pf_cross,
            'loss_pf_inner': loss_pf_inner,
            'loss_ie_fake': loss_ie_fake,
            'loss_ie_real': loss_ie_real,
            'loss_IR_real': loss_IR_real,
            'loss_IR_fake': loss_IR_fake
        }
        
        return loss_dict
        # endregion

    def get_test_loss(self, data_dict: dict, pred_dict: dict) -> dict:
        loss = torch.tensor(0.0).cuda()
        loss_ce_real = torch.tensor(0.0).cuda()
        loss_ce_fake = torch.tensor(0.0).cuda()
        loss_ce = torch.tensor(0.0).cuda()
        loss_ME = torch.tensor(0.0).cuda()
        loss_IR = torch.tensor(0.0).cuda()
        loss_IGR = torch.tensor(0.0).cuda()
        loss_dict = {'total': loss,'loss_bce': loss_ce, 'loss_ce_real': loss_ce_real, 'loss_ce_fake': loss_ce_fake, 'loss_ME': loss_ME, 'loss_IR': loss_IR, 'loss_IGR': loss_IGR}
        return loss_dict

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        if 'source_index' in data_dict and -1 not in data_dict['env_id']: # depend on the dataset for io
            return self.get_train_loss(data_dict,pred_dict)
        else:
            return self.get_test_loss(data_dict, pred_dict)
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data~
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # 1. get the explicit identity
        with torch.no_grad():
            if hasattr(self, 'explicit_extractor') and self.explicit_extractor is not None:
                resized_images = F.interpolate(data_dict['image'], size=(112, 112), mode='bilinear', align_corners=False)
                id_feat = self.explicit_extractor(resized_images)
            else:
                id_feat = self.get_em_feature(data_dict)

        # 2. use xception to get the mixed Identity
        features = self.features(data_dict)
        mixedIdentity = self.backbone.getEmbeddings(features)
        if not inference and 'source_index' in data_dict and 'target_index' in data_dict and 'env_id' in data_dict and 'label' in data_dict:
            with torch.no_grad():
                self.update_im_kernel(
                    mixedIdentity.detach(), 
                    data_dict['target_index'], 
                    data_dict['env_id'], 
                    data_dict['label']
                )

        # 3. use IR module to get the weaked Identity
        if not inference and 'source_index' in data_dict and 'target_index' in data_dict and 'env_id' in data_dict:
            source_index = data_dict['source_index']
            target_index = data_dict['target_index']
            label = data_dict['label']
            env_id = data_dict['env_id']
            weakedIdentity, sourceIdentity, targetIdentity = self.get_weaked_identity(mixedIdentity, source_index, target_index, env_id, label)
        else:
            batch_size = mixedIdentity.size(0)
            feature_dim = mixedIdentity.size(1)
            device = mixedIdentity.device
            empty_f_remove = torch.zeros(batch_size, feature_dim, device=device)
            weakedIdentity = self.ir_module(mixedIdentity, empty_f_remove)  
        
        # 4. use IGR module to purify the final residual identity
        if not inference and 'target_index' in data_dict and 'env_id' in data_dict:
            target_index = data_dict['target_index']
            env_id = data_dict['env_id']
            puredIdentity = self.igr_module.get_pured_identity(weakedIdentity, target_index, env_id, self.em_kernel)
        else:
            batch_size = weakedIdentity.size(0)
            feature_dim = weakedIdentity.size(1)
            device = weakedIdentity.device
            empty_target = torch.zeros(batch_size, feature_dim, device=device)
            puredIdentity = self.igr_module(weakedIdentity, empty_target)


        pred = self.classifier(puredIdentity, id_feat)
        residual_identity = puredIdentity
        fake_prob = torch.softmax(pred, dim=1)[:, 1]

        # 5. build the prediction dictionary, include the explicit identity, purified residual identity, mixed identity, and weak identity
        if not inference and 'source_index' in data_dict and 'target_index' in data_dict and 'env_id' in data_dict:
            pred_dict = {
                'cls': pred,                      # classification result [batch_size, 2]
                'prob': fake_prob,                # fake probability [batch_size]
                'feat': features,                 # feature [batch_size, feature_dim, H, W]
                'id_feat': id_feat,               # explicit identity [batch_size, feature_dim]
                'embed': residual_identity,       # residual identity [batch_size, feature_dim]
                'mixed_identity': mixedIdentity,  # mixed identity [batch_size, feature_dim]
                'weaked_identity': weakedIdentity, # weak identity [batch_size, feature_dim]
                'source_identity': sourceIdentity, # source identity [batch_size, feature_dim]
                'target_identity': targetIdentity # target identity [batch_size, feature_dim]
            }
        else:
            pred_dict = {
                'cls': pred,
                'prob': fake_prob,
                'feat': features, 
                'id_feat': id_feat,
                'embed': residual_identity,
                'mixed_identity': mixedIdentity,
                'weaked_identity': weakedIdentity
            }
        
        return pred_dict
