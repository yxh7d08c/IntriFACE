'''
# author: Yuchen Hui
# email: huiyc@stu.xidian.edu.cn
# date: 2025-10-11
# description: 检测器提取身份与表层身份相似度
'''

import os
import sys
import argparse
import logging
import random
from typing import List

import numpy as np
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


def load_intriFace_model(cfg: dict, checkpoint_path: str | None, device: torch.device, detector_py_path: str):
    import importlib.util

    module_name = 'detectors._runtime_detector_cos'
    spec = importlib.util.spec_from_file_location(module_name, detector_py_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    DetectorClass = getattr(module, 'IntriFace')
    model = DetectorClass(cfg)

    if checkpoint_path and str(checkpoint_path).lower() != 'none':
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()
    return model


def build_loader(dataset_name: str, base_cfg: dict, batch_size: int, num_workers: int, split: str, shuffle: bool, seed: int) -> DataLoader:
    cfg = dict(base_cfg)
    if split == 'train':
        cfg['train_dataset'] = [dataset_name]
    else:
        cfg['test_dataset'] = dataset_name
    logger.info(f"Data split={split}, dataset={dataset_name}")
    mode = split
    dataset = IntriFaceDataset(config=cfg, mode=mode)
    generator = None
    if shuffle:
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )
    return loader


@torch.no_grad()
def collect_cosine_similarities(model, loader: DataLoader, device: torch.device, max_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    similarities: List[float] = []
    labels: List[int] = []
    count_real, count_fake = 0, 0

    for _, batch in enumerate(loader):
        for key in list(batch.keys()):
            if batch[key] is not None and key != 'name':
                batch[key] = batch[key].to(device)

        pred = model(batch, inference=True)
        residual_feat = pred['embed']
        explicit_feat = pred['id_feat']

        cos_sim = F.cosine_similarity(residual_feat, explicit_feat, dim=1)
        cos_np = cos_sim.detach().cpu().numpy()
        y_np = batch['label'].detach().cpu().numpy()

        for i in range(cos_np.shape[0]):
            is_fake = int(y_np[i] == 1)
            if is_fake == 0 and count_real >= max_per_class:
                continue
            if is_fake == 1 and count_fake >= max_per_class:
                continue
            similarities.append(float(cos_np[i]))
            labels.append(int(y_np[i]))
            if is_fake == 0:
                count_real += 1
            else:
                count_fake += 1

        if count_real >= max_per_class and count_fake >= max_per_class:
            break

    sims = np.array(similarities, dtype=float)
    lbs = np.array(labels, dtype=int)
    return sims, lbs


def plot_cosine_bars(similarities: np.ndarray,
                     labels: np.ndarray,
                     out_png: str,
                     title: str,
                     bin_width: float = 0.02,
                     range_min: float = -1.0,
                     range_max: float = 1.0,
                     stacked: bool = False,
                     header_label: str | None = None):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    real_color = "#2008ff"
    fake_color = "#ff0505"
    sims = np.asarray(similarities, dtype=float)
    lbs = np.asarray(labels, dtype=int)

    data_min = float(np.min(sims)) if sims.size > 0 else -1.0
    data_max = float(np.max(sims)) if sims.size > 0 else 1.0
    eff_min = data_min - 0.1
    eff_max = data_max + 0.1
    if eff_min >= eff_max:
        eff_min, eff_max = -1.0, 1.0
    if bin_width <= 0:
        bin_width = 0.02
    edges = np.arange(eff_min, eff_max + bin_width, bin_width)
    centers = edges[:-1] + bin_width / 2.0
    left_edges = edges[:-1]

    real_mask = (lbs == 0)
    fake_mask = (lbs == 1)

    real_counts, _ = np.histogram(sims[real_mask], bins=edges)
    fake_counts, _ = np.histogram(sims[fake_mask], bins=edges)
    fake_counts = fake_counts.astype(float)

    plt.figure(figsize=(6.2, 4.8), dpi=1200)
    ax = plt.gca()
    ax.set_facecolor('white')
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color('#666666')

    if stacked:
        plt.bar(left_edges, real_counts, width=bin_width, align='edge', color=real_color, alpha=0.7,
                label='Real(0)', edgecolor='none', linewidth=0, snap=True)
        plt.bar(left_edges, fake_counts, width=bin_width, align='edge', bottom=real_counts, color=fake_color, alpha=0.7,
                label='Fake(1)', edgecolor='none', linewidth=0, snap=True)
    else:
        bar_w = bin_width * 0.5
        plt.bar(left_edges, real_counts, width=bar_w, align='edge', color=real_color, alpha=0.7,
                label='Real(0)', edgecolor='none', linewidth=0, snap=True)
        plt.bar(left_edges + bar_w, fake_counts, width=bar_w, align='edge', color=fake_color, alpha=0.7,
                label='Fake(1)', edgecolor='none', linewidth=0, snap=True)

    plt.xlim(eff_min, eff_max)
    plt.xlabel('Cosine Similarity', fontsize=17)
    plt.ylabel('Count', fontsize=17)

    tick_candidates = np.arange(-1.0, 1.0001, 0.2)
    ticks = tick_candidates[(tick_candidates >= eff_min) & (tick_candidates <= eff_max)]
    if ticks.size == 0:
        ticks = np.linspace(eff_min, eff_max, num=6)
    plt.xticks(ticks, [f'{v:.1f}' for v in ticks], rotation=0, fontsize=10)

    handles_labels = [
        (plt.Rectangle((0, 0), 1, 1, color='#2008ff', alpha=0.7), 'Real'),
        (plt.Rectangle((0, 0), 1, 1, color='#ff0505', alpha=0.7), 'Fake'),
    ]
    handles, labels_txt = zip(*handles_labels)
    legend = ax.legend(
        handles, labels_txt,
        loc='upper right', bbox_to_anchor=(0.98, 0.98),
        frameon=True, fancybox=True, fontsize=12, ncol=1, borderaxespad=0.2
    )
    frame = legend.get_frame()
    frame.set_alpha(0.65)      
    frame.set_facecolor('white') 
    frame.set_edgecolor('#cccccc')
    frame.set_linewidth(0.8)
    if header_label:
        ax.set_title(str(header_label), fontsize=15, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='cosine similarity histogram')
    parser.add_argument('--detector_config', type=str, required=False,
                        default=os.path.join(PROJECT_ROOT, 'config/detector/intriface.yaml'))
    parser.add_argument('--extra_test_config', type=str, required=False,
                        default=os.path.join(PROJECT_ROOT, 'config/test_config.yaml'))
    parser.add_argument('--checkpoint', type=str, required=False, default='')
    parser.add_argument('--detector_py_path', type=str, required=False,
                        default=os.path.join(PROJECT_ROOT, 'detectors/intriFace_detector.py'))
    parser.add_argument('--dataset', type=str, required=False, default='FF-DF')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='The data division used')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--max_per_class', type=int, default=3000, help='The maximum number of samples per class (too many bars will affect readability)')
    parser.add_argument('--output', type=str, default=os.path.join(PROJECT_ROOT, 'vis/cosine/ffdf-intriFace.png'))
    args = parser.parse_args()

    if args.seed is None or args.seed < 0:
        args.seed = int.from_bytes(os.urandom(4), byteorder='little')
    set_seed(args.seed)

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

    model = load_intriFace_model(cfg, args.checkpoint, device, args.detector_py_path)
    loader = build_loader(args.dataset, cfg, args.batch_size, args.num_workers, split=args.split, shuffle=False, seed=args.seed)

    sims, lbs = collect_cosine_similarities(model, loader, device, max_per_class=args.max_per_class)

    title = f'Cosine Similarity (residual vs explicit) | {args.dataset} [{args.split}] | N={len(sims)} | seed={args.seed}'
    plot_cosine_bars(sims, lbs, args.output, title,
                     header_label=args.dataset)


if __name__ == '__main__':
    main()


