from .vmamba import VSSM, SimpleFlowExtractor
import torch
from torch import nn
import torch.nn.functional as F
from configs.config_setting import setting_config

class FlowVM_Net(nn.Module):
    config = setting_config()
    def __init__(self, input_channels, num_classes, num_frames, depths, depths_decoder, drop_path_rate, load_ckpt_path=None):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.input_channels = setting_config.input_channels
        self.num_classes = setting_config.num_classes
        self.depths = setting_config.depths
        self.depths_decoder = setting_config.depths_decoder
        self.drop_path_rate = setting_config.drop_path_rate
        self.load_ckpt_path = setting_config.load_ckpt_path
        self.num_frames = setting_config.num_frames

        gpu_id = setting_config.gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        self.conv = nn.Conv2d(in_channels = self.num_frames * 3 , out_channels=3, kernel_size=3, padding=1, bias=False)
        self.batchnorm2d = nn.BatchNorm2d(self.input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.flow_feature_extractor = SimpleFlowExtractor().to(self.device)
        self.flowvmnet = VSSM(in_chans=input_channels, num_classes=num_classes, depths=depths, depths_decoder=depths_decoder, drop_path_rate=drop_path_rate).to(self.device)  # 将模型移动到 GPU

    def forward(self, x):
        x = x.to(self.device)
        flow_features = None

        if x.size(1) == self.input_channels * self.num_frames + 2:
            image_features = x[:, :self.input_channels * self.num_frames, :, :]
            flow_features = x[:, self.input_channels * self.num_frames:, :, :]
            image_features = self.conv(image_features)
            image_features = self.batchnorm2d(image_features)
            x = self.relu(image_features)
            flow_features = self.flow_feature_extractor(flow_features)
        elif x.size(1) == 3:
            pass
        else:
            raise ValueError(f"Unexpected number of channels: {x.size(1)}")

        logits = self.flowvmnet(x, flow_features)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.flowvmnet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']

            new_dict = {k: v for k, v in pretrained_dict.items() if
                        k in model_dict.keys() and model_dict[k].size() == v.size()}
            model_dict.update(new_dict)

            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))

            self.flowvmnet.load_state_dict(model_dict, strict=False)  # 使用 strict=False 允许部分加载

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.flowvmnet.state_dict()
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
            self.flowvmnet.load_state_dict(model_dict, strict=False)  # 使用 strict=False 允许部分加载

            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")