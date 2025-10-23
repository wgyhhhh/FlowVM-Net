import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from custom import calc_flow, flow_to_image
import cv2
import torch.nn as nn
from custom import process_and_generate_flow_vector
from configs.config_setting import setting_config

class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, model, args, num_frames, num_classes, train=True, device='cuda', Test=False):
        super(NPY_datasets, self).__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.model = model.to(device)
        self.args = args
        self.device = device

        def valid_files(path):
            return sorted(
                [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and not f.startswith('.')])

        if train:
            images_list = valid_files(path_Data + 'train/images/')
            masks_list = valid_files(path_Data + 'train/masks/')
            self.data = []
            for i in range(0, len(images_list) - num_frames + 1, num_frames):
                img_paths = [path_Data + 'train/images/' + images_list[j] for j in range(i, i + num_frames)]
                mask_path = path_Data + 'train/masks/' + masks_list[i + num_frames - 1]
                self.data.append([img_paths, mask_path])
            self.transformer = config.train_transformer
        else:
            if Test:
                images_list = valid_files(path_Data + 'test/images/')
                masks_list = valid_files(path_Data + 'test/masks/')
                self.data = []
                for i in range(0, len(images_list) - num_frames + 1, num_frames):
                    img_paths = [path_Data + 'test/images/' + images_list[j] for j in range(i, i + num_frames)]
                    mask_path = path_Data + 'test/masks/' + masks_list[i + num_frames - 1]
                    self.data.append([img_paths, mask_path])
                self.transformer = config.test_transformer
            else:
                images_list = valid_files(path_Data + 'val/images/')
                masks_list = valid_files(path_Data + 'val/masks/')
                self.data = []
                for i in range(0, len(images_list) - num_frames + 1, num_frames):
                    img_paths = [path_Data + 'val/images/' + images_list[j] for j in range(i, i + num_frames)]
                    mask_path = path_Data + 'val/masks/' + masks_list[i + num_frames - 1]
                    self.data.append([img_paths, mask_path])
                self.transformer = config.test_transformer

    def __getitem__(self, indx):
        img_paths, msk_path = self.data[indx]

        imgs = [np.array(Image.open(img_path).convert('RGB')) for img_path in img_paths]
        img = np.concatenate(imgs, axis=2)
        img1, img2 = img_paths[0], img_paths[1]

        dataset_type = 'train' if 'train' in img1 else 'val' if 'val' in img1 else 'test'
        flow_path = f"cached_flow/{dataset_type}_{os.path.basename(img1)}_{os.path.basename(img2)}.npy"
        flow_dir = os.path.dirname(flow_path)
        if not os.path.exists(flow_dir):
            os.makedirs(flow_dir)

        if os.path.exists(flow_path):
            flow_tensor = np.load(flow_path)
            flow_tensor = torch.tensor(flow_tensor, dtype=torch.float32).to(self.device)
        else:
            flow_tensor = process_and_generate_flow_vector(self.model, self.args, img1, img2)
            np.save(flow_path, flow_tensor.cpu().numpy())
            flow_tensor = flow_tensor.to(self.device)

        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).to(self.device) / 255.0

        fused_input = torch.cat((img_tensor, flow_tensor), dim=0)
        fused_img = fused_input.permute(1, 2, 0).cpu().detach().numpy()
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255

        fused_img, msk = self.transformer((fused_img, msk))

        return fused_img, msk

    def __len__(self):
        return len(self.data)

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).to(self.device)  # 移动到 GPU
        label = torch.from_numpy(label.astype(np.float32)).to(self.device)  # 移动到 GPU
        sample = {'image': image, 'label': label.long()}
        return sample
