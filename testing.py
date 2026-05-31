import argparse
import os
import sys
import warnings

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from core.raft import RAFT
from core.utils.utils import load_ckpt
from configs.config_setting import json_to_args, setting_config
from dataset import NPY_datasets
from engine import test_one_epoch
from models.flowvmnet import FlowVM_Net
from utils import get_logger, log_config_info, set_seed

warnings.filterwarnings("ignore")


def parse_test_args():
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('--data_path', type=str, default=None, help='Path to dataset root')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='FlowVM-Net checkpoint path')
    parser.add_argument('--batch_size', type=int, default=1, help='Test batch size')
    parser.add_argument('--num_frames', type=int, default=None, help='Number of frames per sample')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID')
    parser.add_argument('--work_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--cfg', default='./configs/spring-M.json', type=str, help='RAFT config file path')
    parser.add_argument('--model', default='./pre_trained_weights/Tartan-C-T-TSKH-spring540x960-M.pth',
                        type=str, help='RAFT checkpoint path')
    parser.add_argument('--save_predictions', action='store_true', help='Save predicted masks')
    return parser.parse_args()


def apply_cli_overrides(config, args):
    for name in ('data_path', 'num_frames', 'num_classes', 'gpu_id', 'work_dir'):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    return config


def resolve_device(gpu_id):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(gpu_id)}")
    return torch.device("cpu")


def save_predictions(model, dataloader, device, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, _) in enumerate(dataloader):
            images = images.to(device, non_blocking=True).float()
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            save_path = os.path.join(save_dir, f"pred_{idx:04d}.png")
            vutils.save_image(outputs.clamp(0, 1), save_path)
            print(f"Saved prediction: {save_path}")


def main(config):
    args = parse_test_args()
    apply_cli_overrides(config, args)
    device = resolve_device(config.gpu_id)

    raft_args = json_to_args(args.cfg)
    raft_args.model = args.model
    raft_args.device = str(device)
    raft_model = RAFT(raft_args)
    load_ckpt(raft_model, raft_args.model)
    raft_model = raft_model.to(device).eval()

    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    outputs = os.path.join(config.work_dir, 'outputs')
    os.makedirs(outputs, exist_ok=True)

    logger = get_logger('test', log_dir)
    writer = SummaryWriter(os.path.join(config.work_dir, 'summary'))
    log_config_info(config, logger)

    set_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_dataset = NPY_datasets(
        config.data_path, config, raft_model, raft_args,
        config.num_frames, config.num_classes, train=False, device=device, Test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=config.num_workers,
        drop_last=False,
    )

    model = FlowVM_Net(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        num_frames=config.num_frames,
        depths=config.depths,
        depths_decoder=config.depths_decoder,
        drop_path_rate=config.drop_path_rate,
        load_ckpt_path=config.load_ckpt_path,
    )
    model.load_from()
    model = model.to(device)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    best_weight = torch.load(args.checkpoint_path, map_location='cpu')
    if isinstance(best_weight, dict) and 'model_state_dict' in best_weight:
        best_weight = best_weight['model_state_dict']
    model.load_state_dict(best_weight, strict=False)

    criterion = config.criterion
    loss = test_one_epoch(test_loader, model, criterion, logger, config, device)
    print(f"Test completed. Loss: {loss:.4f}")

    if args.save_predictions:
        pred_save_dir = os.path.join(config.work_dir, "predictions")
        save_predictions(model, test_loader, device, pred_save_dir)
        print(f"Predictions saved to {pred_save_dir}")

    writer.close()


if __name__ == '__main__':
    config = setting_config
    main(config)
