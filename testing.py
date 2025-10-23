import torch
from torch.utils.data import DataLoader
import timm
from dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.flowvmnet import FlowVM_Net
from configs.config_setting import parse_args
from engine import *
import os
import sys
from core.raft import RAFT
from utils import *
from configs.config_setting import setting_config
import argparse
import warnings
from core.utils.utils import load_ckpt
import torchvision.utils as vutils   # ✅ 用于保存图片
warnings.filterwarnings("ignore")


def save_predictions(model, dataloader, device, save_dir):
    """
    在测试过程中保存预测结果到 save_dir
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)

            # 如果输出是字典或多个值，取第一个
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            # 保存预测图片 (归一化到 [0,1])
            save_path = os.path.join(save_dir, f"pred_{idx:04d}.png")
            vutils.save_image(outputs.sigmoid(), save_path)  # sigmoid 适合二分类/掩码输出
            print(f"Saved prediction: {save_path}")


def main(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./configs/spring-M.json', type=str)
    parser.add_argument('--model', default='./pre_trained_weights/Tartan-C-T-TSKH-spring540x960-M.pth', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--checkpoint_dir',
                        default='/home/test3/test3/test3/wgy/FlowVM-Net-main/results/_vessel_Thursday_11_September_2025_20h_02m_46s/checkpoints',
                        type=str)
    raft_args = parse_args(parser)
    raft_model = RAFT(raft_args)
    load_ckpt(raft_model, raft_args.model)
    device = torch.device('cuda' if raft_args.device == 'cuda' else 'cpu')
    raft_model = raft_model.to(device).eval()

    # log / outputs
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.test_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    os.makedirs(outputs, exist_ok=True)

    global logger
    logger = get_logger('test', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    # dataset
    val_dataset = NPY_datasets(config.data_path, config, raft_model, raft_args, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    # model
    model_cfg = config.model_config
    model = FlowVM_Net(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )
    model.load_from()

    device = torch.device('cuda:1')
    model = model.to(device)

    # load weights
    best_model_path = '/home/test3/test3/test3/wgy/FlowVM-Net-main/results/_vessel_Monday_15_September_2025_14h_43m_29s/checkpoints/best.pth'
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    best_weight = torch.load(best_model_path, map_location='cpu')
    model.load_state_dict(best_weight, strict=False)

    # test
    criterion = config.criterion
    loss = test_one_epoch(val_loader, model, criterion, logger, config, device)
    print(f"Test completed. Loss: {loss:.4f}")

    # 保存预测结果
    pred_save_dir = os.path.join(config.work_dir, "predictions")
    save_predictions(model, val_loader, device, pred_save_dir)
    print(f"Predictions saved to {pred_save_dir}")


if __name__ == '__main__':
    config = setting_config
    main(config)
