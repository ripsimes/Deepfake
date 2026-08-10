"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset
from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict
import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


import json
from PIL import Image
from torchvision import transforms

class FlatJsonTestDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, config):
        with open(json_path) as f:
            self.items = json.load(f)
        res = config['resolution']
        self.transform = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])
        # what test_epoch expects:
        self.data_dict = {'image': [it['path'] for it in self.items]}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it['path']).convert('RGB')
        return {
            'image': self.transform(img),
            'label': torch.tensor(it['label']).long(),
            'mask': None,
            'landmark': None,
        }

    @staticmethod
    def collate_fn(batch):
        return {
            'image':    torch.stack([b['image'] for b in batch]),
            'label':    torch.stack([b['label'] for b in batch]),
            'mask':     None,
            'landmark': None,
        }


def prepare_testing_data(config):
    json_path = config.get('test_json_path',
                           "/content/DiffusionForensics_Recons/test.json")
    print(f"[TEST] loading flat JSON: {json_path}")
    test_set = FlatJsonTestDataset(json_path, config)
    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config['test_batchSize'],
        shuffle=False,
        num_workers=int(config['workers']),
        collate_fn=test_set.collate_fn,
        drop_last=False,
    )
    return {config['test_dataset'][0]: loader}


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())
    
    np.save("/content/preds.npy", np.array(prediction_lists))
    np.save("/content/labels.npy", np.array(label_lists))
    return np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)
    
def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps,feat_nps = test_one_dataset(model, test_data_loaders[key])
        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    config['lmdb'] = False
    print("DEBUG lmdb flag:", config.get('lmdb'), "| dataset_json_folder:", config.get('dataset_json_folder'))

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)

        # unwrap if it's a training-state dict (Recce / ForensicHub style)
        if isinstance(ckpt, dict) and "model" in ckpt and not any(
            k.endswith(".weight") or k.endswith(".bias") for k in ckpt.keys()
        ):
            ckpt = ckpt["model"]
        
        # strip "module." prefix from DDP-saved checkpoints
        if any(k.startswith("module.") for k in ckpt.keys()):
            ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        
        # Try a normal load first (Effort, Capsule-Net, SPSL, DeepfakeBench-trained Recce)
        try:
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(f"[load:flat] missing={len(missing)}  unexpected={len(unexpected)}")
        except (RuntimeError, AttributeError):
            # Fallback: ForensicHub-trained Recce, where weights live under model.model.* / model.backbone.*
            inner = {k[len("model."):]: v for k, v in ckpt.items() if k.startswith("model.")}
            bb    = {k.replace("model.encoder.", "", 1): v
                     for k, v in ckpt.items() if k.startswith("model.encoder.")}
        
            m_missing, m_unexp = model.model.load_state_dict(inner, strict=False)
            if hasattr(model, "backbone"):
                b_missing, b_unexp = model.backbone.load_state_dict(bb, strict=False)
                print(f"[load:split] inner miss={len(m_missing)} bbone miss={len(b_missing)}")
            else:
                print(f"[load:split] inner miss={len(m_missing)} unexp={len(m_unexp)}")
        
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders)
    print('===> Test Done!')

if __name__ == '__main__':
    main()
