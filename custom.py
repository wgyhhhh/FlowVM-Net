import sys
sys.path.append('core')
import argparse
import os
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import datasets
from raft import RAFT
from core.utils.flow_viz import flow_to_image
from core.utils.utils import load_ckpt

def json_to_args(json_path):
    # return a argparse.Namespace object
    with open(json_path, 'r') as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args

def parse_args(parser):
    entry = parser.parse_args()
    json_path = entry.cfg
    args = json_to_args(json_path)
    args_dict = args.__dict__
    for index, (key, value) in enumerate(vars(entry).items()):
        args_dict[key] = value
    return args

def forward_flow(args, model, image1, image2):
    device = next(model.parameters()).device
    image1 = image1.to(device)
    image2 = image2.to(device)

    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final



def calc_flow(args, model, image1, image2):
    device = next(model.parameters()).device
    image1 = image1.to(device)
    image2 = image2.to(device)

    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (
                0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down


@torch.no_grad()
def process_and_generate_flow(model, args, image1_path, image2_path, output_path='./custom/', device=torch.device('cuda')):
    os.makedirs(output_path, exist_ok=True)

    image1 = cv2.imread(image1_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread(image2_path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)

    H, W = image1.shape[1:]

    image1 = image1[None].to(device)
    image2 = image2[None].to(device)

    flow, info = calc_flow(args, model, image1, image2)

    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)

    cv2.imwrite(f"{output_path}flow.jpg", flow_vis)

    flow_tensor = torch.tensor(flow_vis).permute(2, 0, 1).float() / 255.0

    return flow_tensor


@torch.no_grad()
def process_and_generate_flow_vector(model, args, image1_path, image2_path, device=torch.device('cuda')):
    """
    直接返回光流二维向量，而不是可视化图像
    """
    image1 = cv2.imread(image1_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread(image2_path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)

    image1 = image1[None].to(device)
    image2 = image2[None].to(device)

    flow, info = calc_flow(args, model, image1, image2)

    return flow[0]

@torch.no_grad()
def process_images(model, args, image1_path, image2_path):

    image1 = cv2.imread(image1_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread(image2_path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1) / 255.0
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1) / 255.0

    device = next(model.parameters()).device
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)

    flow_tensor = demo_data('./nijiushige/', args, model, image1, image2)

    return flow_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', default='./config/eval/spring-M.json', type=str)
    parser.add_argument('--model', help='checkpoint path',default= './pretrain/Tartan-C-T-TSKH-spring540x960-M.pth', type=str)
    parser.add_argument('--device', help='inference device', type=str, default='cpu')
    args = parse_args(parser)
    model = RAFT(args)
    load_ckpt(model, args.model)
    if args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    demo_custom(model, args, device=device)

if __name__ == '__main__':
    main()