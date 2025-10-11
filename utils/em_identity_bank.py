'''
# author: Yuchen Hui
# email: huiyc@stu.xidian.edu.cn
# date: 2025-10-11
# description: 表层身份库的构建
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import glob
from collections import defaultdict
import sys
import re
import cv2

sys.path.append('.../IntriFACE')
from models.encoder128 import Backbone128, load_custom_state_dict
from models.iib import IIB

class CelebDFDataset(Dataset):
    def __init__(self, image_paths, transform=None, transform_albu=None):
        self.image_paths = image_paths
        self.transform = transform
        self.transform_albu = transform_albu
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform_albu:
            image_np = np.array(image)
            transformed = self.transform_albu(image=image_np)
            image = transformed['image']
        
        if self.transform:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image = self.transform(image)
            
        return image, img_path

def get_em_feature(encoder, iib, images, device, N, param_dict):
    with torch.no_grad():
        images = images.to(device)
        X_id = encoder(
            F.interpolate(images, size=[128, 128],
                        mode='bilinear', align_corners=True),
            cache_feats=True
        )
        
        # 01 Get Inter-features After One Feed-Forward:
        min_std = torch.tensor(0.01, device=device)
        readout_feats = [(encoder.features[i] - param_dict[i][0]) / torch.max(param_dict[i][1], min_std)
                        for i in range(N + 1)]

        # 02 information restriction:
        X_id_restrict = torch.zeros_like(X_id).to(device)
        for i in range(N):
            R = encoder.features[i] 
            Z, lambda_, _ = getattr(iib.module if hasattr(iib, 'module') else iib, f'iba_{i}')(
                R, readout_feats,
                m_r=param_dict[i][0], std_r=param_dict[i][1],
                active_neurons=param_dict[i][2],
            )
            X_id_restrict += encoder.restrict_forward(Z, i)
        X_id_restrict /= float(N)
        
        encoder.features = []

    return X_id_restrict

def count_images_in_directory(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    count = 0
    for ext in image_extensions:
        count += len(glob.glob(os.path.join(directory, f'*{ext}')))
    return count

def parse_real_dir(dir_name):
    match = re.match(r'id(\d+)_(\d+)', dir_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def main():
    parser = argparse.ArgumentParser(description='Build explicit identity bank for Celeb-DF-v2 dataset')
    parser.add_argument('--dataset_path', type=str, default='.../datasets/rgb/Celeb-DF-v2', 
                        help='Path to the Celeb-DF-v2 dataset')
    parser.add_argument('--output_path', type=str, default='explicit_identity_bank.pth', 
                        help='Path to save the explicit identity bank')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for feature extraction')
    parser.add_argument('--encoder_path', type=str, default='.../IntriFACE/models/model_128_ir_se50.pth', 
                        help='Path to the encoder model')
    parser.add_argument('--ib_mode', type=str, default='smooth', choices=['smooth', 'no_smooth'],
                        help='IB mode: smooth or no_smooth')
    parser.add_argument('--checkpoints_dir', type=str, default='.../IntriFACE/base',
                        help='Directory containing checkpoints')
    parser.add_argument('--weights_dir', type=str, default='.../IntriFACE/models/weights128_celeb', 
                        help='Directory containing readout layer weights')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--use_train_augmentation', action='store_true', default=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Stage 1: Counting real images and creating structure...")
    
    real_images_dir = os.path.join(args.dataset_path, 'Celeb-real', 'frames')
    real_identity_structure = {}
    
    for dir_name in os.listdir(real_images_dir):
        dir_path = os.path.join(real_images_dir, dir_name)
        if os.path.isdir(dir_path):
            person_id, env_id = parse_real_dir(dir_name)
            if person_id is not None:
                if person_id not in real_identity_structure:
                    real_identity_structure[person_id] = {}
                
                num_images = count_images_in_directory(dir_path)
                real_identity_structure[person_id][env_id] = num_images
    
    structure_info = {
        'real_identity_structure': real_identity_structure
    }
    
    torch.save(structure_info, 'celeb_df_real_structure_info.pth')
    print("Structure information saved to celeb_df_real_structure_info.pth")
    
    print("Stage 2: Initializing models...")
    
    N = 10
    encoder = Backbone128(50, 0.6, 'ir_se').eval().to(device)
    original_state_dict = torch.load(args.encoder_path, map_location=device)
    load_custom_state_dict(encoder, original_state_dict)
    
    with torch.no_grad():
        _ = encoder(torch.rand(1, 3, 128, 128).to(device), cache_feats=True)
        _readout_feats = encoder.features[:(N + 1)]
        in_c = sum(map(lambda f: f.shape[-3], _readout_feats))
        out_c_list = [_readout_feats[i].shape[-3] for i in range(N)]
        encoder.features = [] 
    
    ROOT = {
        'smooth': {'root': os.path.join(args.checkpoints_dir, 'w_kernel_smooth'), 'path': 'ckpt_ks_I.pth'},
        'no_smooth': {'root': os.path.join(args.checkpoints_dir, 'wo_kernel_smooth'), 'path': 'ckpt_I.pth'}
    }
    
    root = ROOT[args.ib_mode]['root']
    pathI = ROOT[args.ib_mode]['path']
    
    iib = IIB(in_c, out_c_list, device, smooth=(args.ib_mode=='smooth'), kernel_size=1).eval().to(device)
    iib_path = os.path.join(root, pathI)
    
    if os.path.exists(iib_path):
        print(f"Loading IIB model from {iib_path}")
        iib.load_state_dict(torch.load(iib_path, map_location=device), strict=(args.ib_mode=='smooth'))
    else:
        print(f"Warning: IIB model path {iib_path} not found. Using randomly initialized model.")
    
    param_dict = []
    for i in range(N + 1):
        weight_path = os.path.join(args.weights_dir, f'readout_layer{i}.pth')
        if os.path.exists(weight_path):
            state = torch.load(weight_path, map_location=device)
            n_samples = state['n_samples'].float()
            std = torch.sqrt(state['s'] / (n_samples - 1)).to(device)
            neuron_nonzero = state['neuron_nonzero'].float()
            active_neurons = (neuron_nonzero / n_samples) > 0.01
            param_dict.append([state['m'].to(device), std, active_neurons.to(device)])
        else:
            param_dict.append([torch.zeros(1).to(device), torch.ones(1).to(device), torch.ones(1, dtype=torch.bool).to(device)])

    
    import albumentations as A
    from dataset.albu import IsotropicResize
    
    if args.use_train_augmentation:
        transform_albu = A.Compose([           
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=[-10, 10], p=0.5),
            A.GaussianBlur(blur_limit=[3, 7], p=0.5),
            IsotropicResize(max_side=256, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=[-0.1, 0.1], contrast_limit=[-0.1, 0.1]),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5)
        ])
    else:
        transform_albu = A.Compose([           
            IsotropicResize(max_side=256, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        ])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    em_kernel = {}
    missing_ids = set() 
    
    for person_id in tqdm(real_identity_structure.keys(), desc="Processing real identities"):
        person_features = {}
        
        for env_id in real_identity_structure[person_id].keys():
            dir_path = os.path.join(real_images_dir, f"id{person_id}_{env_id:04d}")
            image_paths = []
            
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.extend(glob.glob(os.path.join(dir_path, f'*{ext}')))
            
            if not image_paths:
                print(f"Warning: No images found in {dir_path}")
                missing_ids.add(person_id)
                continue
                
            dataset = CelebDFDataset(image_paths, transform=transform, transform_albu=transform_albu)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                                   num_workers=args.num_workers, pin_memory=True)
            
            all_features = []
            for images, _ in dataloader:
                features = get_em_feature(encoder, iib, images, device, N, param_dict)
                all_features.append(features.cpu())
            
            if all_features:
                all_features = torch.cat(all_features, dim=0)
                person_features[env_id] = all_features
            else:
                missing_ids.add(person_id)
        
        if person_features:
            max_env_id = max(person_features.keys()) + 1
            feature_dim = next(iter(person_features.values())).size(1)
            
            all_env_features = torch.zeros(feature_dim, max_env_id)
            for env_id, features in person_features.items():
                all_env_features[:, env_id] = features.mean(dim=0)
            
            em_kernel[person_id] = all_env_features
        else:
            missing_ids.add(person_id)
    
    for person_id in real_identity_structure.keys():
        if person_id not in em_kernel:
            missing_ids.add(person_id)
    
    explicit_identity_bank = {
        'em_kernel': em_kernel,
        'missing_ids': list(missing_ids) 
    }
    
    torch.save(explicit_identity_bank, args.output_path)

if __name__ == "__main__":
    main()