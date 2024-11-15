import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
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
warnings.filterwarnings("ignore")


def main(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', default='./configs/spring-M.json', type=str)
    parser.add_argument('--model', help='checkpoint path', default='./pre_trained_weights/Tartan-C-T-TSKH-spring540x960-M.pth',
                        type=str)
    parser.add_argument('--device', help='inference device', type=str, default='cpu')
    parser.add_argument('--checkpoint_dir', help='directory to load checkpoints from', default='./results/vmunet_isic18_Saturday_12_October_2024_13h_33m_07s/checkpoints',
                        type=str)
    raft_args = parse_args(parser)
    raft_model = RAFT(raft_args)
    load_ckpt(raft_model, raft_args.model)
    if raft_args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    raft_model = raft_model.to(device)
    raft_model.eval()

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.test_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('test', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    val_dataset = NPY_datasets(config.data_path, config, raft_model, raft_args, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Preparing Model----------#')
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


    model = model.cuda()

    print('#----------Loading Model Weights for Testing----------#')
    best_model_path = os.path.join(checkpoint_dir, 'best.pth')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    best_weight = torch.load(best_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(best_weight, strict=False)

    print('#----------Testing----------#')
    criterion = config.criterion
    loss = test_one_epoch(
        val_loader,
        model,
        criterion,
        logger,
        config,
    )

    print(f"Test completed. Loss: {loss:.4f}")


if __name__ == '__main__':
    config = setting_config
    main(config)
