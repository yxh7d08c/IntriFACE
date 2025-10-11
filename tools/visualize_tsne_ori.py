'''
# author: Yuchen Hui
# email: huiyc@stu.xidian.edu.cn
# date: 2025-10-11
# description: backbone对真伪的区分程度
'''

import os
import sys
import argparse
import logging
import random

import numpy as np
import yaml

import torch
from openTSNE import TSNE


from torch.utils.data import DataLoader
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PROJECT_ROOT = '.../IntriFACE'
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from dataset.intriFace_dataset import IntriFaceDataset


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_config(base_cfg: dict, extra_cfg: dict) -> dict:
    cfg = dict(base_cfg or {})
    for k, v in (extra_cfg or {}).items():
        cfg[k] = v
    return cfg


def _abs_from_project_root(path_value: str) -> str:
    if isinstance(path_value, str) and not os.path.isabs(path_value):
        return os.path.normpath(os.path.join(PROJECT_ROOT, path_value.lstrip('./')))
    return path_value



def build_loader(dataset_name: str, base_cfg: dict, batch_size: int, num_workers: int, seed: int, split: str) -> DataLoader:
    cfg = dict(base_cfg)
    if split == 'train':
        cfg['train_dataset'] = [dataset_name]
    else:
        cfg['test_dataset'] = dataset_name
    dataset = IntriFaceDataset(config=cfg, mode=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )
    return loader


def load_intriFace_model(cfg: dict, device: torch.device, detector_py_path: str):
    import importlib.util
    module_name = 'detectors._runtime_detector_visual'
    spec = importlib.util.spec_from_file_location(module_name, detector_py_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    DetectorClass = getattr(module, 'IntriFace')
    model = DetectorClass(cfg)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_features(model, loader: DataLoader, device: torch.device, paths_iter=None):
    features, labels, names = [], [], []

    for _, batch in enumerate(loader):
        for key in list(batch.keys()):
            if batch[key] is not None and key != 'name':
                batch[key] = batch[key].to(device)
        pred = model(batch, inference=True)

        residual = pred['embed']  # [B, C]
        id_feat = pred['id_feat']  # [B, C]
        diff_feat = (residual - id_feat).detach().cpu().numpy()

        y = batch['label'].detach().cpu().numpy()  # 0: real, 1: fake
        name_list = None
        if isinstance(batch, dict):
            if 'name' in batch:
                name_list = batch['name']
            elif 'path' in batch:
                name_list = batch['path']
        if name_list is None:
            if paths_iter is not None:
                name_list = list(itertools.islice(paths_iter, diff_feat.shape[0]))
            else:
                name_list = [None] * diff_feat.shape[0]
        try:
            name_list = [str(n) if n is not None else None for n in name_list]
        except Exception:
            name_list = [None] * diff_feat.shape[0]

        for i in range(diff_feat.shape[0]):
            features.append(diff_feat[i])
            labels.append(int(y[i]))
            names.append(name_list[i])

    X = np.stack(features, axis=0)
    y = np.array(labels)
    return X, y, names


def _video_id_from_name(name: str) -> str:
    if name is None:
        return 'unknown'
    return os.path.dirname(name)


def aggregate_per_video(X: np.ndarray, y: np.ndarray, names: list):
    groups = {}
    for i, n in enumerate(names):
        vid = _video_id_from_name(n)
        groups.setdefault(vid, []).append(i)

    X_agg, y_agg = [], []
    for idxs in groups.values():
        feats = X[idxs]
        labs = y[idxs]
        X_agg.append(feats.mean(axis=0))
        y_agg.append(int(np.round(labs.mean())))

    if len(X_agg) <= 1 and X.shape[0] > 1:
        return X, y
    X_agg = np.stack(X_agg, axis=0)
    y_agg = np.array(y_agg, dtype=int)
    return X_agg, y_agg

from openTSNE import TSNE

def run_tsne(X: np.ndarray, use_opentsne: bool, perplexity: float, n_iter: int, random_state: int, n_jobs: int = 8) -> np.ndarray:
    if use_opentsne:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=True,
        )
        embedding = tsne.fit(X)
        return embedding
    else:
        from sklearn.manifold import TSNE as SKTSNE
        tsne = SKTSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            init='pca',
            learning_rate='auto'
        )
        return tsne.fit_transform(X)


def plot_and_save(embedding: np.ndarray, labels: np.ndarray, out_png: str, title: str = 't-SNE of (residual - id_feat)'):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.rcParams['axes.edgecolor'] = '#aaaaaa'
    plt.rcParams['axes.linewidth'] = 0.6
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    color_map = np.where(labels == 1, '#2008ff', '#ff0505')

    plt.figure(figsize=(6.8, 6.8), dpi=220)
    plt.scatter(
        embedding[:, 0], embedding[:, 1],
        c=color_map,
        s=14,  
        alpha=0.5, 
        linewidths=0, 
        marker='o'
    )
    plt.xticks([])
    plt.yticks([])
    import matplotlib.lines as mlines
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization of IntriFace features (residual - id_feat)')
    parser.add_argument('--detector_config', type=str, required=False,
                        default=os.path.join(PROJECT_ROOT, 'config/detector/intriface.yaml'))
    parser.add_argument('--extra_test_config', type=str, required=False,
                        default=os.path.join(PROJECT_ROOT, 'config/test_config.yaml'))
    parser.add_argument('--detector_py_path', type=str, required=False,
                        default=os.path.join(PROJECT_ROOT, 'detectors/intriFace_detector.py'))
    parser.add_argument('--dataset', type=str, required=False, default='Celeb-DF-v2', help='Visualization dataset name')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--output', type=str, default=os.path.join(PROJECT_ROOT, 'tools/celeb/celebv2-ori.png'))
    parser.add_argument('--max_per_class', type=int, default=4000, help='The maximum number of samples per class (only effective when not aggregated)')
    parser.add_argument('--aggregate', type=str, choices=['none', 'video'], default='none', help='Whether to aggregate by video (mean of video frames)')
    parser.add_argument('--split', type=str, choices=['train','val','test'], default='train', help='Use data subset, default train')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--use_opentsne', action='store_true')
    parser.add_argument('--perplexity', type=float, default=20.0)
    parser.add_argument('--n_iter', type=int, default=1500)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)
    det_cfg = load_yaml(args.detector_config)
    extra_cfg = load_yaml(args.extra_test_config) if os.path.isfile(args.extra_test_config) else {}
    cfg = merge_config(det_cfg, extra_cfg)
    if 'dataset_json_folder' in cfg:
        cfg['dataset_json_folder'] = _abs_from_project_root(cfg['dataset_json_folder'])
    if 'rgb_dir' in cfg:
        cfg['rgb_dir'] = _abs_from_project_root(cfg['rgb_dir'])
    if 'lmdb_dir' in cfg:
        cfg['lmdb_dir'] = _abs_from_project_root(cfg['lmdb_dir'])

    model = load_intriFace_model(cfg, device, args.detector_py_path)
    loader = build_loader(args.dataset, cfg, args.batch_size, args.num_workers, seed=args.seed, split=args.split)


    if args.aggregate == 'video':
        paths_list = []
        try:
            paths_list = loader.dataset.data_dict.get('image', [])
        except Exception:
            paths_list = []
        X_frames, y_frames, names = collect_features(
            model,
            loader,
            device,
            paths_iter=iter(paths_list),
        )
        X, y = aggregate_per_video(X_frames, y_frames, names)
        try:
            n_real = int((y == 0).sum())
            n_fake = int((y == 1).sum())
        except Exception:
            pass
    else:
        paths_list = []
        try:
            paths_list = loader.dataset.data_dict.get('image', [])
        except Exception:
            paths_list = []
        X_frames, y_frames, _ = collect_features(
            model,
            loader,
            device,
            paths_iter=iter(paths_list),
        )
        if args.max_per_class is not None and args.max_per_class > 0:
            idx_real = np.where(y_frames == 0)[0][:args.max_per_class]
            idx_fake = np.where(y_frames == 1)[0][:args.max_per_class]
            idx = np.concatenate([idx_real, idx_fake])
            X, y = X_frames[idx], y_frames[idx]
        else:
            X, y = X_frames, y_frames


    Z = run_tsne(X, use_opentsne=args.use_opentsne, perplexity=args.perplexity, n_iter=args.n_iter, random_state=args.seed)

    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)
    plot_and_save(
        Z,
        y,
        args.output,
        title=f't-SNE (residual - id_feat) | {args.dataset} | N={X.shape[0]} | agg={args.aggregate} | perplexity={args.perplexity} | seed={args.seed}'
    )


if __name__ == '__main__':
    main()


