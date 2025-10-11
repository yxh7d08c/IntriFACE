'''
# author: Yuchen Hui
# email: huiyc@stu.xidian.edu.cn
# date: 2025-10-11
# description: 检测器性能指标
'''

import os
import sys
import copy
import json
import argparse
import logging
import yaml
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import importlib.util
PROJECT_ROOT = '.../IntriFACE'
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from dataset.intriFace_dataset import IntriFaceDataset
from metrics.utils import get_test_metrics, parse_metric_for_print


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_config(base_cfg: dict, extra_cfg: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    if extra_cfg:
        for k, v in extra_cfg.items():
            cfg[k] = v
    return cfg


def build_test_loader_for_dataset(dataset_name: str, base_cfg: dict, batch_size: int, num_workers: int) -> DataLoader:
    cfg = copy.deepcopy(base_cfg)
    cfg['test_dataset'] = dataset_name
    mode = 'test'
    dataset = IntriFaceDataset(config=cfg, mode=mode)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )
    return loader


@torch.no_grad()
def model_inference(model, data_dict: dict, device: torch.device):
    for key in list(data_dict.keys()):
        if data_dict[key] is not None and key != 'name':
            data_dict[key] = data_dict[key].to(device)
    model.eval()
    return model(data_dict, inference=True)


def load_intriFace_model(cfg: dict, checkpoint_path: str, device: torch.device, detector_py_path: str):
    module_name = 'detectors._runtime_detector'
    spec = importlib.util.spec_from_file_location(module_name, detector_py_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    DetectorClass = getattr(module, 'IntriFace')
    model = DetectorClass(cfg)
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


def evaluate_one_dataset(model, loader: DataLoader, device: torch.device, pos_label: int = 1):
    preds, labels = [], []
    for _, batch in tqdm(enumerate(loader), total=len(loader), desc='Eval', ncols=100):
        predictions = model_inference(model, batch, device)
        preds += list(predictions['prob'].detach().cpu().numpy())
        labels += list(batch['label'].detach().cpu().numpy())
    preds_np = np.array(preds)
    labels_np = np.array(labels)
    img_names = loader.dataset.data_dict['image']
    metric = get_test_metrics(y_pred=preds_np, y_true=labels_np, img_names=img_names, pos_label=pos_label)
    return metric


def main():
    parser = argparse.ArgumentParser(description='independent test script (based on IntriFaceDataset and JSON labels)')
    parser.add_argument('--detector_config', type=str, required=False,
                        default=os.path.join(PROJECT_ROOT, 'config/detector/intriface.yaml'),
                        help='Detector configuration file path')
    parser.add_argument('--extra_test_config', type=str, required=False,
                        default=os.path.join(PROJECT_ROOT, 'config/test_config.yaml'),
                        help='Test additional configuration (dataset_json_folder、label_dict etc.)')
    parser.add_argument('--checkpoint', type=str, required=True, help='IntriFace weight path')
    parser.add_argument('--test_datasets', type=str, nargs='*', default='FF-DF',
                        help='List of datasets to test; empty uses test_dataset from detector_config')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, 'test/metrics'))
    parser.add_argument('--auc_positive_class', type=str, choices=['fake','real'], default='fake',
                        help='Positive class for AUC calculation: fake(=1) or real(=0), default fake')
    parser.add_argument('--detector_py_path', type=str, required=False,
                        default=os.path.join(PROJECT_ROOT, 'detectors/intriFace_detector.py'))
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)
    pos_label = 1 if args.auc_positive_class == 'fake' else 0
    det_cfg = load_yaml(args.detector_config)
    extra_cfg = load_yaml(args.extra_test_config) if os.path.isfile(args.extra_test_config) else {}
    cfg = merge_config(det_cfg, extra_cfg)
    def _abs_from_project_root(path_value: str) -> str:
        if isinstance(path_value, str) and not os.path.isabs(path_value):
            return os.path.normpath(os.path.join(PROJECT_ROOT, path_value.lstrip('./')))
        return path_value

    if 'dataset_json_folder' in cfg:
        cfg['dataset_json_folder'] = _abs_from_project_root(cfg['dataset_json_folder'])
    if 'rgb_dir' in cfg:
        cfg['rgb_dir'] = _abs_from_project_root(cfg['rgb_dir'])
    if 'lmdb_dir' in cfg:
        cfg['lmdb_dir'] = _abs_from_project_root(cfg['lmdb_dir'])

    test_datasets = args.test_datasets if args.test_datasets else cfg.get('test_dataset', [])
    if isinstance(test_datasets, str):
        test_datasets = [test_datasets]
    model = load_intriFace_model(cfg, args.checkpoint, device, args.detector_py_path)
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_all = {}
    avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0, 'video_auc': 0, 'dataset_dict': {}}
    metric_scoring = cfg.get('metric_scoring', 'auc')

    for name in test_datasets:
        loader = build_test_loader_for_dataset(name, cfg, args.batch_size, args.num_workers)
        metric = evaluate_one_dataset(model, loader, device, pos_label=pos_label)
        metrics_all[name] = metric
        with open(os.path.join(args.output_dir, f'metrics_{name}.json'), 'w') as f:
            json.dump({k: (float(v) if hasattr(v, 'item') else (v.tolist() if hasattr(v, 'tolist') else v))
                       for k, v in metric.items() if k not in ['pred', 'label']}, f, ensure_ascii=False, indent=2)
        for k in ['acc', 'auc', 'eer', 'ap', 'video_auc']:
            avg_metric[k] += float(metric[k])
        avg_metric['dataset_dict'][name] = float(metric.get(metric_scoring, 0.0))

    for k in ['acc', 'auc', 'eer', 'ap', 'video_auc']:
        avg_metric[k] /= len(test_datasets)
    metrics_all['avg'] = avg_metric
    pretty = parse_metric_for_print(metrics_all)
    logger.info(pretty)
    def _to_jsonable(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        return x

    sanitized = {}
    for ds_name, metric in metrics_all.items():
        if ds_name == 'avg':
            sanitized['avg'] = {}
            for k, v in metric.items():
                if k == 'dataset_dict' and isinstance(v, dict):
                    sanitized['avg'][k] = {dk: _to_jsonable(dv) for dk, dv in v.items()}
                else:
                    sanitized['avg'][k] = _to_jsonable(v)
        else:
            sanitized[ds_name] = {k: _to_jsonable(v) for k, v in metric.items() if k not in ['pred', 'label']}

    with open(os.path.join(args.output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(sanitized, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()


