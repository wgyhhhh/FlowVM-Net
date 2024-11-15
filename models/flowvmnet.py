from .vmamba import VSSM, FlowFeatureExtractor
import torch
from torch import nn
import torch.nn.functional as F

class FlowVM_Net(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 2, 2],
                 depths_decoder=[2, 2, 2, 1],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 device='cuda'  # 添加设备参数
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.device = device  # 保存设备信息
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1))
        self.batchnorm2d = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.flow_feature_extractor = FlowFeatureExtractor().to(device)  # 将模型移动到 GPU
        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           ).to(device)  # 将模型移动到 GPU

    def forward(self, x):
        x = x.to(self.device)  # 将输入移动到 GPU
        flow_features = None  # Initializes flow_features to None by default.

        if x.size(1) == 9:
            image_features = x[:, :6, :, :]
            flow_features = x[:, 6:, :, :]
            image_features = image_features.view(x.size(0), 3, 2, x.size(2), x.size(3))
            image_features = self.conv3d(image_features)
            image_features = image_features.squeeze(2)
            image_features = self.batchnorm2d(image_features)
            x = self.relu(image_features)
            flow_features = self.flow_feature_extractor(flow_features)

        elif x.size(1) == 3:
            pass  # No need to do anything.
        else:
            raise ValueError(f"Unexpected number of channels: {x.size(1)}")

        logits = self.vmunet(x, flow_features)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']

            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if
                        k in model_dict.keys() and model_dict[k].size() == v.size()}
            model_dict.update(new_dict)

            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))

            # 加载更新后的状态字典
            self.vmunet.load_state_dict(model_dict, strict=False)  # 使用 strict=False 允许部分加载

            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            # 处理 decoder 部分
            model_dict = self.vmunet.state_dict()
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v

            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if
                        k in model_dict.keys() and model_dict[k].size() == v.size()}
            model_dict.update(new_dict)

            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))

            # 加载更新后的状态字典
            self.vmunet.load_state_dict(model_dict, strict=False)  # 使用 strict=False 允许部分加载

            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")