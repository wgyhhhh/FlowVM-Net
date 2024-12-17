# FlowVM-Net

|<img align="left" width="900" height="700" src="https://github.com/wgyhhhh/FlowVM-Net/blob/main/images234.jpg">|
|:--:|
| **FlowVM-Net** |

### Abstract
Precise segmentation of continuous vessels from X-ray coronary angiography (XCA) image sequences is pivotal in advancing the diagnosis and treatment of coronary artery disease. However, motion artifacts and shadowing in XCA images significantly complicate the segmentation of continuous vessel structures. This paper proposes a novel encoder-decoder deep network architecture called FlowVM-Net, which focuses on the current frame while leveraging multiple contextual frames to segment 2D vessel masks. The network incorporates an Selective Scan Module 2D (SSM-2D) module and a multi-layer dilated convolution interaction mechanism into the Wavelet Dilated Convolution Vision State Space Mamba (WD-VSM) block to effectively capture a broader range of contextual information without increasing computational costs. To efficiently discriminate vessel features from low-contrast images and dynamic background artifacts, we employ a multi-scale channel attention feature fusion mechanism between the final encoder layer and the initial decoder layer to combine down-sampled features with optical flow information. This approach complements the dynamic information in the XCA image sequences, leading to improved segmentation results. Additionally, we use a composite loss function with a BIoU loss function to train the network, improving the accuracy of vessel edge segmentation. Experimental results demonstrate that FlowVM-Net outperforms state-of-the-art methods in addressing the low boundary segmentation quality in coronary artery images, achieving notable improvements in evaluation metrics and exhibiting superior performance.

**0. Environments.**
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
The .whl files of mamba_ssm could be found [here](https://pan.baidu.com/s/1VY19t3dstzWAOXtkRHqHJg?pwd=cfrn).

**1. Datasets.**

The file format reference is as follows. (The image is a 24-bit png image. The mask is an 8-bit png image. (0 pixel dots for background, 255 pixel dots for target))
- './data/your_dataset/'
  - train
    - images
      - 001.png
      - 002.png
      - ...
    - masks
      - 001.png
      - 002.png
      - ...
  - val
    - images
      - 001.png
      - 002.png
      - ...
    - masks
      - 001.png
      - 002.png
      - ...
  - test
      - images
        - 001.png
        - 002.png
        - ...
      - masks
        - 001.png
        - 002.png
        - ...
        - 
**2. Train the FlowVM-Net.**
- The weights of the pre-trained VMamba could be downloaded [here](https://drive.usercontent.google.com/download?id=1uUPsr7XeqayCxlspqBHbg5zIWx0JYtSX&export=download&authuser=0&confirm=t&uuid=8f3d1bcd-cd88-4ca1-a758-7049c1ebc144&at=AN_67v29VPGaI2TjZsEPsB3Z7y3h%3A1727950609222).
- The weights of the pre-trained Optical Flow model could be downloaded [here](https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW).
After that, the pre-trained weights should be stored in './pre_trained_weights/'.
```
python train.py
```
- After trianing, you could obtain the outputs in `` ./results/``
  
**3. Test the FlowVM-Net.**
First, in the testing.py file, you should change the address of the checkpoint in 'checkpoint path'.
```
python testing.py
```
