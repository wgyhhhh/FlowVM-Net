import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from custom import process_and_generate_flow_vector


IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, model, args, num_frames, num_classes,
                 train=True, device='cuda', Test=False):
        super(NPY_datasets, self).__init__()
        self.root = os.path.normpath(path_Data)
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.model = model.to(device)
        self.args = args
        self.device = torch.device(device)
        self.flow_cache_dir = getattr(config, 'flow_cache_dir', os.path.join(self.root, 'cached_flow'))

        split = self._resolve_split(train=train, test=Test)
        self.data = self._build_samples(split)
        self.transformer = config.train_transformer if train else config.test_transformer

        if not self.data:
            raise RuntimeError(
                f"No samples found for split '{split}' under {self.root}. "
                "Expected either training/validation/test or train/val/test with images and labels/masks."
            )

    def _resolve_split(self, train, test):
        if train:
            candidates = ('training', 'train')
        elif test:
            candidates = ('test',)
        else:
            candidates = ('validation', 'val')

        for split in candidates:
            if os.path.isdir(os.path.join(self.root, split, 'images')):
                return split
        return candidates[0]

    def _split_dirs(self, split):
        split_dir = os.path.join(self.root, split)
        image_dir = os.path.join(split_dir, 'images')
        for label_name in ('labels', 'masks'):
            label_dir = os.path.join(split_dir, label_name)
            if os.path.isdir(label_dir):
                return image_dir, label_dir
        return image_dir, os.path.join(split_dir, 'labels')

    @staticmethod
    def _valid_files(path):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        return sorted(
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
            and not f.startswith('.')
            and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )

    @staticmethod
    def _sequence_key(filename):
        stem = os.path.splitext(filename)[0]
        match = re.match(r'(.+)_i(\d+)$', stem)
        if match:
            return match.group(1), int(match.group(2))
        return stem, 0

    def _label_lookup(self, labels):
        return {name.lower(): name for name in labels}

    def _frame_label_path(self, label_dir, label_lookup, target_image_name):
        stem, ext = os.path.splitext(target_image_name)
        candidate_names = [
            target_image_name,
            target_image_name.replace('image_', 'label_', 1),
        ]

        for candidate in candidate_names:
            match = label_lookup.get(candidate.lower())
            if match is not None:
                return os.path.join(label_dir, match)

        return None

    def _sequence_label_path(self, label_dir, label_lookup, sequence_key):
        candidate_names = [
            f'{sequence_key}.png',
            f'{sequence_key}.jpg',
            f'{sequence_key}.jpeg',
            f'{sequence_key}.tif',
            f'{sequence_key}.tiff',
            f'{sequence_key.replace("image_", "label_", 1)}.png',
            f'{sequence_key.replace("image_", "label_", 1)}.jpg',
            f'{sequence_key.replace("image_", "label_", 1)}.jpeg',
            f'{sequence_key.replace("image_", "label_", 1)}.tif',
            f'{sequence_key.replace("image_", "label_", 1)}.tiff',
        ]

        for candidate in candidate_names:
            match = label_lookup.get(candidate.lower())
            if match is not None:
                return os.path.join(label_dir, match)

        return None

    def _build_samples(self, split):
        image_dir, label_dir = self._split_dirs(split)
        images = self._valid_files(image_dir)
        labels = self._valid_files(label_dir)
        label_lookup = self._label_lookup(labels)

        grouped = defaultdict(list)
        for image_name in images:
            sequence_key, frame_index = self._sequence_key(image_name)
            grouped[sequence_key].append((frame_index, image_name))

        samples = []
        for sequence_key in sorted(grouped):
            frames = [name for _, name in sorted(grouped[sequence_key])]
            if len(frames) < self.num_frames:
                continue

            sequence_label = self._sequence_label_path(label_dir, label_lookup, sequence_key)
            if sequence_label is not None:
                last_window = frames[-self.num_frames:]
                img_paths = [os.path.join(image_dir, name) for name in last_window]
                samples.append((split, img_paths, sequence_label))
                continue

            for start in range(0, len(frames) - self.num_frames + 1, self.num_frames):
                window = frames[start:start + self.num_frames]
                target_image_name = window[-1]
                img_paths = [os.path.join(image_dir, name) for name in window]
                label_path = self._frame_label_path(label_dir, label_lookup, target_image_name)
                if label_path is None:
                    raise FileNotFoundError(
                        f"Could not find frame label for image '{target_image_name}' in {label_dir}."
                    )
                samples.append((split, img_paths, label_path))

        return samples

    def _flow_cache_path(self, split, img1, img2):
        os.makedirs(self.flow_cache_dir, exist_ok=True)
        model_id = os.path.splitext(os.path.basename(getattr(self.args, 'model', 'raft')))[0]
        name = f"{split}_{model_id}_{os.path.basename(img1)}_{os.path.basename(img2)}.npy"
        return os.path.join(self.flow_cache_dir, name)

    def __getitem__(self, indx):
        split, img_paths, msk_path = self.data[indx]

        imgs = [np.array(Image.open(img_path).convert('RGB')) for img_path in img_paths]
        img = np.concatenate(imgs, axis=2)
        img1, img2 = img_paths[0], img_paths[1]

        flow_path = self._flow_cache_path(split, img1, img2)
        if os.path.exists(flow_path):
            flow_tensor = torch.from_numpy(np.load(flow_path)).float()
        else:
            flow_tensor = process_and_generate_flow_vector(
                self.model, self.args, img1, img2, device=self.device
            ).detach().cpu().float()
            np.save(flow_path, flow_tensor.numpy())

        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        fused_input = torch.cat((img_tensor, flow_tensor), dim=0)
        fused_img = fused_input.permute(1, 2, 0).numpy()
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255.0

        fused_img, msk = self.transformer((fused_img, msk))
        return fused_img, msk

    def __len__(self):
        return len(self.data)
