'''
# author: Yuchen Hui
# email: huiyc@stu.xidian.edu.cn
# date: 2025-10-11
# description: 预处理——分布权重生成
'''

import os
import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import argparse
import glob
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder128 import Backbone128, load_custom_state_dict


class CelebDFDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        person_dirs = glob.glob(os.path.join(root_dir, "id*_*"))
        for person_dir in tqdm(person_dirs, desc="Scanning person directories"):
            for ext in ['.png', '.jpg', '.jpeg']:
                self.image_paths.extend(glob.glob(os.path.join(person_dir, f"*{ext}")))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image


def compute_layer_statistics(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Backbone128(50, 0.6, 'ir_se').eval().to(device)
    encoder_path = os.path.join(args.model_dir, 'model_128_ir_se50.pth')
    original_state_dict = torch.load(encoder_path, map_location=device)
    load_custom_state_dict(encoder, original_state_dict)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = CelebDFDataset(
        root_dir=args.data_root,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    N = 10 
    statistics = []
    for _ in range(N + 1):
        statistics.append({
            'm_sum': 0,
            's_sum': 0,
            'neuron_active': 0, 
            'n_samples': 0 
        })
    
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Processing batches"):
            images = images.to(device)
            face_region = F.interpolate(images, 
                                      size=[128, 128],
                                      mode='bilinear', 
                                      align_corners=True)

            _ = encoder(face_region, cache_feats=True)
            
            for i in range(N + 1):
                feat_flat = encoder.features[i]
                batch_size = feat_flat.size(0)
                
                statistics[i]['m_sum'] += feat_flat.sum(0)
                statistics[i]['s_sum'] += (feat_flat ** 2).sum(0)
                
                statistics[i]['neuron_active'] += (feat_flat.abs() > 0.01).sum(0)
                statistics[i]['n_samples'] += batch_size
            
            encoder.features = []
    
    for i in range(N + 1):
        n_samples = statistics[i]['n_samples']
        m = statistics[i]['m_sum'] / n_samples
        s = statistics[i]['s_sum'] 
        neuron_nonzero = statistics[i]['neuron_active']
        
        state = {
            'm': m.cpu(), 
            's': s.cpu(),
            'n_samples': torch.tensor(n_samples),
            'neuron_nonzero': neuron_nonzero.cpu()
        }
        
        save_path = os.path.join(args.output_dir, f'readout_layer{i}.pth')
        torch.save(state, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute statistics of Celeb-DF-v2 real face images')
    parser.add_argument('--data_root', type=str, 
                        default='.../datasets/rgb/Celeb-DF-v2/Celeb-real/frames')
    parser.add_argument('--model_dir', type=str, 
                        default='.../IntriFACE/models')
    parser.add_argument('--output_dir', type=str, 
                        default='.../IntriFACE/models/weights128_celeb')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    compute_layer_statistics(args)