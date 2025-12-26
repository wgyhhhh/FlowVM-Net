import torch
from torch.utils.data import DataLoader
import timm
import os
import sys
from core.raft import RAFT
from utils import *
from configs.config_setting import setting_config
import argparse
import warnings
from core.utils.utils import load_ckpt
from dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.flowvmnet import FlowVM_Net
from configs.config_setting import json_to_args
from engine import *
from argparse import Namespace

warnings.filterwarnings("ignore")

def parse_all_args():
    """解析所有训练参数"""
    parser = argparse.ArgumentParser(description='Training script')

    # Training Parameters
    parser.add_argument('--data_path', type=str, default=None, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--num_frames', type=int, default=None, help='Number of frames')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID') 
    parser.add_argument('--work_dir', type=str, default=None, help='Working directory')
    
    # RAFT Parameters 
    parser.add_argument('--cfg', type=str, default='./configs/spring-M.json', 
                       help='RAFT config file path')
    parser.add_argument('--model', type=str, 
                       default='./pre_trained_weights/Tartan-C-T-TSKH-spring540x960-M.pth', 
                       help='RAFT model checkpoint path')

    return parser.parse_args()


def main(config):
    # Parse all arguments
    args = parse_all_args()
    print('#----------Preparing Flow Generation Module----------#')

    # FlowVM-Net config args
    config.data_path = args.data_path
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.num_frames = args.num_frames
    config.num_classes = args.num_classes
    config.gpu_id = args.gpu_id
    device_str = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    # RAFT model args
    raft_args = json_to_args(args.cfg)
    raft_args.model = args.model
    raft_args.device = device_str

    # Initialize RAFT model
    raft_model = RAFT(raft_args)
    load_ckpt(raft_model, raft_args.model)

    # Set device for RAFT model
    gpu_id = int(config.gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    raft_model = raft_model.to(device)
    raft_model.eval()

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    num_frames = config.num_frames
    num_classes = config.num_classes
    train_dataset = NPY_datasets(config.data_path, config, raft_model, raft_args, num_frames, num_classes, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, raft_model, raft_args, num_frames, num_classes, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)
    test_dataset = NPY_datasets(config.data_path, config, raft_model, raft_args, num_frames, num_classes, train=False,
                                Test=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=config.num_workers,
                             drop_last=True)

    print('#----------Prepareing Model----------#')

    model = FlowVM_Net(
        num_classes=config.num_classes,
        input_channels=setting_config.input_channels,
        num_frames=config.num_frames,
        depths=setting_config.depths,
        depths_decoder=setting_config.depths_decoder,
        drop_path_rate=setting_config.drop_path_rate,
        load_ckpt_path=setting_config.load_ckpt_path,
    )
    model.load_from()
    gpu_id = int(config.gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        print(f'#---------- Epoch {epoch} ----------#')
        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            device
        )

        print('#----------Validation----------#')
        loss = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config,
            device
        )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
            device
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    main(config)