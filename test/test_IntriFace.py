'''
# author: Yuchen Hui
# email: huiyc@stu.xidian.edu.cn
# date: 2025-10-11
# description: 检测器判别性能
'''

import os
import sys

import argparse
import logging
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
sys.path.append('.../IntriFACE')
from detectors.intriFace_detector import IntriFace
from detectors import DETECTOR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.image_files.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {
                'image': image,
                'path': image_path
            }
        except Exception as e:
            if self.transform:
                empty_image = torch.zeros((3, 256, 256))
            else:
                empty_image = np.zeros((256, 256, 3), dtype=np.uint8)
            return {
                'image': empty_image,
                'path': image_path
            }

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_transform():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform

@torch.no_grad()
def inference(model, data_dict, device):
    data_dict['image'] = data_dict['image'].to(device)
    model.eval()
    predictions = model(data_dict, inference=True)
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='test IntriFace model')
    parser.add_argument('--config', type=str, required=False, default='.../IntriFACE/config/detector/intriface.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint path')
    parser.add_argument('--image_dir', type=str, required=True, help='image directory path')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--output', type=str, default='results.txt', help='result output file')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for fake detection')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    device = torch.device(args.device)
    config = load_config(args.config)
    transform = get_transform()
    dataset = TestDataset(args.image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = IntriFace(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    results = []
    for batch in tqdm(dataloader, desc="Testing"):
        predictions = inference(model, batch, device)
        fake_probs = predictions['prob'].cpu().numpy()
        for i, path in enumerate(batch['path']):
            fake_prob = fake_probs[i]
            is_fake = fake_prob >= args.threshold
            results.append({
                'path': path,
                'fake_prob': fake_prob,
                'is_fake': is_fake
            })
    
    with open(args.output, 'w') as f:
        f.write("Image Path,Fake Probability,Result\n")
        for result in results:
            f.write(f"{result['path']},{result['fake_prob']:.6f},{result['is_fake']}\n")
    

    fake_count = sum(1 for result in results if result['is_fake'])
    real_count = len(results) - fake_count

if __name__ == '__main__':
    main()
