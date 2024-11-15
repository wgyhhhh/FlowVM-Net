# FlowVM-Net

## 0. Main Environments
```bash
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

## 1. Prepare the dataset

*D. Prepare your own dataset* </br>

The file format reference is as follows. (The image is a 24-bit png image. The mask is an 8-bit png image. (0 pixel dots for background, 255 pixel dots for target))

- './data/your_dataset/'
  - train
    - images
      - train_001.png
      - train_002.png
      - ...
    - masks
      - train_001.png
      - train_002.png
      - ...
  - val
    - images
      - val_001.png
      - val_002.png
      - ...
    - masks
      - val_001.png
      - val_002.png
      - ...
  - test
      - images
        - test_001.png
        - test_002.png
        - ...
      - masks
        - test_001.png
        - test_002.png
        - ...

## 2. Train the Flow VM-Net
```
python train.py
```
- After trianing, you could obtain the outputs in './results/' </br>
