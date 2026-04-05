# IntriFace: Modeling Intrinsic Forgery Identity through Latent Representation for Generalizable Deepfake Detection

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![PyTorch](https://img.shields.io/badge/Pytorch-2.6.0-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10.16-brightgreen)

<b> Authors: Yuchen Hui, Hanyi Wang, Ming Zhang, Jiayu Guo, Zhipeng He, Qingqwen Li, <a href='https://scholar.google.com/citations?hl=zh-CN&user=oEcRS84AAAAJ&view_op=list_works&sortby=pubdate'>Hui Li*</a> </b>

---
## 📚 **Overview**
<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="./framework.jpg" style="max-width:99%;">
</div>

Welcome to IntriFace, a powerful framework for deepfake face-swapping detection that builds a reliable security barrier between you and the virtual world. Below are its core highlights:

> 📌 **More Reasonable Detection Focus**: Instead of relying on visual or frequency-domain artifacts, IntriFace takes the identity consistency difference between real and fake faces as the core detection basis. This ensures strong generalization, even on unseen datasets and diffusion-model-generated forgeries.
>
> 📌 **Robust Perceptual Interference Suppression**: Equipped with a dedicated encoder, it extracts identity information while suppressing irrelevant interference, focusing on core identity features to avoid misdetection.
>
> 📌 **Advanced Feature Purification**: Its specially designed purification network filters redundant noise and amplifies key discriminative features, expanding the feature gap between real and fake faces.
>
> 📌 **Excellent Comprehensive Performance**: Experiments show IntriFace outperforms other SOTA methods in generalization and robustness. It can accurately recover residual identity information, clearly model identity consistency differences, and demonstrate great potential as a universal detection solution.

---

<font size=4><b> Table of Contents </b></font>

- [Quick Start](#-quick-start)
  - [Environmental](#1-Environmental)
  - [Download Data](#2-download-data)
  - [Preprocessing](#3-preprocessing)
  - [Training](#4-Training)
  - [Evaluation](#5-evaluation)
- [Results](#-results)
- [Citation](#-citation)
- [Copyright](#%EF%B8%8F-license)

---

## ⏳ Quick Start

### 1. Environmental
<a href="#top">[Back to top]</a>

You can run the following script to configure the necessary environment:

```
git clone git@github.com:huiyuchen708/IntriFACE.git
cd IntriFACE
conda create -n IntriFace python=3.10.16
conda activate IntriFace
pip install -r requirements.txt
```

### 2. Download Data
<a href="#top">[Back to top]</a>

All datasets used in IntriFace can be downloaded from their corresponding original repositories. For convenience, we provide a subset of the datasets employed in our study (the remaining ones are sourced from existing works). Each provided dataset has been preprocessed into aligned facial frames (32 frames per video) along with corresponding masks and landmarks, allowing others to directly deploy these faces for evaluating IntriFace. 

The download links and detailed information for each dataset are summarized below:

| Dataset | Real Videos | Fake Videos | Total Videos/Images | Rights Cleared | Total Subjects | Synthesis Methods | Original Repository |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FaceForensics++ | 1000 | 4000 | 5000 | NO | N/A | 4 | [Hyper-link](https://github.com/ondyari/FaceForensics/tree/master/dataset) |
| FaceShifter | 1000 | 1000 | 2000 | NO | N/A | 1 | [Hyper-link](https://github.com/ondyari/FaceForensics/tree/master/dataset) |
| Deepfake Detection Challenge (Preview) | 1131 | 4119 | 5250 | YES | 66 | 2 | [Hyper-link](https://ai.facebook.com/datasets/dfdc/) |
| Deepfake Detection Challenge | 23654 | 104500 | 128154 | YES | 960 | 8 | [Hyper-link](https://www.kaggle.com/c/deepfake-detection-challenge/data) |
| CelebDF-v1 | 408 | 795 | 1203 | NO | N/A | 1 | [Hyper-link](https://github.com/yuezunli/celeb-deepfakeforensics/tree/master/Celeb-DF-v1) |
| CelebDF-v2 | 590 | 5639 | 6229 | NO | 59 | 1 | [Hyper-link](https://github.com/yuezunli/celeb-deepfakeforensics/tree/master/Celeb-DF-v2) |
| UADFV | 49 | 49 | 98 | NO | 49 | 1 | [Hyper-link](https://www.kaggle.com/datasets/adityakeshri9234/uadfv-dataset) |
| DF40 | 1590 | 0.1M+ | 0.1M+ | NO | N/A | 40 | [Hyper-link](https://github.com/YZY-stack/DF40) |
| VGGFace | N/A | N/A | 2.6M+ | NO | 2600 | N/A | [Hyper-link](https://github.com/NNNNAI/VGGFace2-HQ) |
| CASIA-FaceV5 | N/A | N/A | 2500 | NO | 600 | N/A | [Hyper-link](https://huggingface.co/datasets/JustinLeee/Cleaned_Augmented_CASIA_FaceV5) |


🛡️ **Copyright of the above datasets belongs to their original providers.**

After downloading, please store the datasets in the `datasets/rgb` directory and organize them according to the following structure. 

```
rgb
├── Celeb-DF-v2
|   ├── Celeb-real
|   |   ├── frames
|   |   |   ├── id0_0000
|   |   |   |   ├── 000.png
|   |   |   |   ├── ...
|   |   |   ├── idx_xxxx
|   |   |   ├── ...
|   |   ├── landmarks
|   |   |   ├── id0_0000
|   |   |   |   ├── 000.npy
|   |   |   ├── idx_xxxx
|   |   |   ├── ...
|   ├── Celeb-synthesis
|   |   ├── frames
|   |   |   ├── id0_id1_0000
|   |   |   |   ├── 000.png
|   |   |   ├── idx_idx_xxxx
|   |   |   ├── ...
|   |   ├── landmarks
|   |   |   ├── id0_id1_0000
|   |   |   |   ├── 000.npy
|   |   |   ├── idx_idx_xxxx
|   |   |   ├── ...
|   ├── YouTube-real
|   |   ├── frames
|   |   |   ├── 00000
|   |   |   |   ├── 000.png
|   |   |   ├── xxxxx
|   |   |   ├── ...
|   |   ├── landmarks
|   |   |   ├── 00000
|   |   |   |   ├── 000.npy
|   |   |   ├── xxxxx
|   |   |   ├── ...
|   ├── List_of_testing_videos.txt
├── UADFV
|   ├── fake
|   |   ├── frames
|   |   |   ├── 0000_fake
|   |   |   |   ├── 000.png
|   |   |   ├── xxxx_fake
|   |   |   ├── ...
|   |   ├── landmarks
|   |   |   ├── 0000_fake
|   |   |   |   ├── 000.npy
|   |   |   ├── xxxx_fake
|   |   |   ├── ...
|   ├── real
|   |   ├── frames
|   |   |   ├── 0000
|   |   |   |   ├── 000.png
|   |   |   ├── xxxx
|   |   |   ├── ...
|   |   ├── landmarks
|   |   |   ├── 0000
|   |   |   |   ├── 000.npy
|   |   |   ├── xxxx
|   |   |   ├── ...
Other datasets are similar to the above structure
```

If you choose to store the datasets elsewhere, you can specify the path via `rgb_dir` in `training/test_config.yaml` and `training/train_config.yaml`. You may also specify a different location for configuration files by setting `dataset_json_folder` in the same configuration files.

### 3. Preprocessing
<a href="#top">[Back to top]</a>

To stabilize data partitioning and labeling, reduce the training cost of IntriFace, and enhance its training efficiency, you should first execute the following command to unify multi-source datasets into an intuitive sample list. This operation also facilitates reproducibility for other researchers.

```
cd IntriFACE/generate_data
python rearrange.py
```

You can specify the datasets to be processed and their root directories in `generate_data/config.yaml`. In addition, when employing the Information Bottleneck component to extract facial identity representations, it is necessary to pre-generate the dataset distribution information to ensure dynamic adjustment of active neurons during training and to regularize the feature outputs of each layer. Please follow the steps below:

1. Navigate to the `IntriFACE/utils` directory and modify the `--data_root` parameter in the `compute_statistics.py` to point to the actual path where the dataset is stored.

2. Execute the `compute_statistics.py` script within the virtual environment IntriFace.

3. The output weight information will be saved to the `IntriFACE/models` directory.

```
cd IntriFACE/utils
python compute_statistics.py --data_root your_dataset_path --batch_size 32
```

### 4. Training
<a href="#top">[Back to top]</a>

To start training, you first need to download the required [pretrained weights](https://drive.google.com/drive/folders/1-TGxK2pwQKez-S7QoYJ8Y3ImXhz1tQiY?usp=sharing). Then, execute the following command to begin training:

```
CUDA_VISIBLE_DEVICES=2,3 nohup torchrun --nproc_per_node=2 train.py > train.log 2>&1 &
```

Specifically, when resuming training, you can use the `--resume_checkpoint` parameter to specify the model state to restore and set the `--resume_mode` parameter to determine whether to start from a new epoch.

### 5. Evaluation
<a href="#top">[Back to top]</a>

Two evaluation scripts are provided to assess the effectiveness of the trained models:  

1. **test_IntriFace.py**: Designed for rapid inference, this script allows you to specify a directory containing facial images to be evaluated (including any custom deepfake content). It returns the detector’s prediction results for each sample.  
2. **test_IntriFace_datasets.py**: Used for comprehensive performance evaluation of the detector, this script reports aggregated metrics such as AUC, ACC, and EER across different test datasets.

In addition, we provide scripts for visualizing Figures 8 and 9 in the paper. Specifically, you can navigate to the `IntriFACE/tools` directory and select the desired script to execute:  

1. **visualize_cosine.py**: Plots the cosine similarity between the latent-space identity representations extracted by IntriFace and the corresponding surface-space identities. You need to specify the `--checkpoint` parameter to point to your pretrained weights and adjust the `--dataset` parameter to select the test set to be processed.  
2. **visualize_tsne_ori.py**: Visualizes the baseline model’s ability to distinguish between real and fake samples. You can modify the `--aggregate` parameter to indicate whether the dimensionality reduction is performed based on video IDs or image IDs.  
3. **visualize_tsne.py**: Visualizes the pretrained model’s discrimination capability between real and fake samples.

## 🏆 Results
<a href="#top">[Back to top]</a>

We demonstrate the outstanding generalization capability of IntriFace. We strongly recommend referring to our paper for a detailed discussion of IntriFace’s performance differences from existing state-of-the-art methods in terms of robustness, residual identity integrity, and identity purity.

<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="./generalization.jpg" style="max-width:90%;">
</div>

These resources provide a detailed analysis of the training outcomes and offer a deeper understanding of the methodology and findings.

## 📝 Citation

<a href="#top">[Back to top]</a>

If you find our benchmark useful to your research, please cite it as follows:

```
coming soon.
```

## 🛡️ License

<a href="#top">[Back to top]</a>

If you have any suggestions, comments, or wish to contribute code or propose methods, we warmly welcome your input. Please contact us at zhangming2025@njust.edu.cn or huiyc@stu.xidian.edu.cn. We look forward to collaborating with you in pushing the boundaries of facial privacy protection.

