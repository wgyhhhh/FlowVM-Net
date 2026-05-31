# FlowVM-Net

This is the official code repository for "FlowVM-Net: Enhanced Vessel Segmentation in X-Ray Coronary Angiography Using Temporal Information Fusion", which is accpeted by **Journal of Imaging Informatics in Medicine** as a original paper!

## Framework Overview

![Framework Architecture](./framework.png)
*Figure 1: The overall architecture of FlowVM-Net combining spatial features with temporal information*

## 0. Environments.
```bash
conda create -n flowvmnet python=3.8
conda activate flowvmnet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs opencv-python
```
The .whl files of mamba_ssm could be found [here](https://github.com/state-spaces/mamba/releases).
The .whl files of causal_conv1d could be found [here](https://github.com/Dao-AILab/causal-conv1d/releases).

## 1. Prepare the dataset.

Data Format

```
├── './data/XCAV_DIAS/'
    ├── training
        ├── images
            ├── image_s000002_i0.png
            ├── image_s000002_i1.png
            └── ... # image_{sequence_id}_i{frame_index}.png
        ├── labels
            ├── image_s000002_i1.png
            └── ... # same filename as target frame is preferred
    ├── validation
        ├── images
        ├── labels
    ├── test
        ├── images
        ├── labels
```

The loader groups frames by the part before `_i{frame_index}`. For example,
`image_s000002_i0.png` and `image_s000002_i1.png` are treated as one temporal
sequence. The label for each sample is matched against the last frame in the
window first, then against sequence-level names such as `image_s000002.png` or
`label_s000002.png`. The old `train/val/test` and `masks` naming scheme is still
supported for compatibility.

## 2. Train the FlowVM-Net.
- The weights of the pre-trained VMamba could be downloaded [here](https://drive.usercontent.google.com/download?id=1uUPsr7XeqayCxlspqBHbg5zIWx0JYtSX&export=download&authuser=0&confirm=t&uuid=8f3d1bcd-cd88-4ca1-a758-7049c1ebc144&at=AN_67v29VPGaI2TjZsEPsB3Z7y3h%3A1727950609222).
- The weights of the pre-trained Optical Flow model could be downloaded [here](https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW).
After that, the pre-trained weights should be stored in './pre_trained_weights/'.

```
bash train.sh
# or
python train.py --data_path /path/to/XCAV_DIAS --gpu_id 0
```
- After trianing, you could obtain the outputs in `` ./results/``
  
## 3. Test the FlowVM-Net.
```
python testing.py --data_path /path/to/XCAV_DIAS --checkpoint_path /path/to/best.pth --gpu_id 0
```

## 4. Citation
Please cite the paper as follows if you use the code from FlowVM-Net:

```bibtex
@article{wei2025flowvm,
  title={FlowVM-Net: Enhanced Vessel Segmentation in X-Ray Coronary Angiography Using Temporal Information Fusion},
  author={Guangyu Wei and Xueying Zeng and Qing Zhang},
  journal={Journal of Imaging Informatics in Medicine},
  year={2025},
  publisher={Springer},
  doi={10.1007/s10278-025-01732-y}
}
```
