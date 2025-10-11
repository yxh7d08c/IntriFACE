import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
class FaceFeatures:
    def __init__(self, model_path, device='cuda'):
       self.device = torch.device(device)
       checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
       self.model = checkpoint['model'] if isinstance(checkpoint, dict) else checkpoint
       self.model = self.model.to(self.device)
       self.model.eval()
       self.normalize = transforms.Normalize(
           mean=[0.485, 0.456, 0.406], 
           std=[0.229, 0.224, 0.225]
       )
       
       self.transform = transforms.Compose([
           transforms.Resize((112, 112)),
           transforms.ToTensor(),
           self.normalize
       ])
    def get_features(self, img):
        with torch.no_grad():
            if isinstance(img, Image.Image):
                img = self.transform(img)
                img = img.unsqueeze(0)
            elif isinstance(img, torch.Tensor):
                if img.ndim == 3:
                    img = img.unsqueeze(0) 
                if img.shape[2:] != (112, 112):
                    img = F.interpolate(img, size=(112, 112), mode='bilinear', align_corners=True)
                if img.max() > 1:
                    img = img / 255.0
                img = self.normalize(img)
            
            img = img.to(self.device)
            features = self.model(img)
            features = F.normalize(features, p=2, dim=1)
            
            return features.cpu()
    def compare_features(self, feat1, feat2):
       cos_similarity = F.cosine_similarity(feat1, feat2)
       return cos_similarity.item()
    @staticmethod
    def load_image(image_path):
       return Image.open(image_path).convert('RGB')