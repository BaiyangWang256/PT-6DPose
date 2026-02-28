# Enhancing Industrial Robotic Grasping through Point Transformer-Based Category-Level 6D Pose Estimation

[cite_start]**PT-6DPose** integrates Point Transformer, Transformer-based NOCS prediction, and cross-modal attention for robust RGB-D fusion, achieving state-of-the-art performance on NOCS-REAL275[cite: 348].

> **📢 Important Note:** > This repository contains the official code and dataset instructions for the manuscript currently submitted and under review at **The Visual Computer**. If you find this codebase, our pre-trained models, or our methodology helpful in your research, we strongly encourage and kindly request that you cite our manuscript (see the Citation section below). 
> 
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx) *(Note: Please replace with your actual Zenodo/GitHub DOI once generated to ensure long-term accessibility)*

## Abstract
In the realm of Industry 4.0 and smart manufacturing, 6D object pose estimation is pivotal for automating tasks such as industrial bin-picking and robotic assembly. Addressing the limitations of existing methods in handling unstructured environments, this paper introduces a novel framework for category-level 6D object pose estimation using Point Transformer. Our approach leverages multi-scale self-attention mechanisms for robust point cloud feature extraction and incorporates a Transformer-based NOCS predictor for stable coordinate prediction. A cross-modal attention mechanism is integrated to fuse RGB and depth features effectively. Experiments on the NOCS-REAL275 dataset demonstrate our method's superiority, achieving an IoU50 of 83.9% and a 5°2cm pose accuracy of 49.5%, outperforming state-of-the-art methods. This framework shows significant potential for real-world industrial robotic applications.

## Citation
If you use our code or models in your research, please cite our paper:
```bibtex
@article{wang2026enhancing,
  title={Enhancing Industrial Robotic Grasping through Point Transformer-Based Category-Level 6D Pose Estimation},
  author={Wang, Baiyang and Wang, Xujian and Fang, Ming and Wang, Hongjun and Li, Hua},
  journal={The Visual Computer},
  note={Under Review},
  year={2026}
}
Environment SettingsThe code has been tested with:python 3.9torch 1.12cuda 11.3DependenciesInstall the required packages. Because our backbone relies on Point Transformer V3, you will also need to configure the Pointcept library.Bashpip install -r requirements.txt
pip install opencv-python

# Install Pointcept dependencies for PTv3
cd model/pointcept
python setup.py install
Data ProcessingNOCS Dataset (CAMERA25 & REAL275)Our framework relies on the standard NOCS benchmarks.Download and preprocess the dataset following the standard NOCS format.Download the instance masks and segmentation results.Organize your data directory as follows:Plaintextdata
├── CAMERA25
│   ├── train
│   └── val
├── REAL275
│   ├── train
│   └── test
├── obj_models
└── segmentation_results
TrainTo train the model on the NOCS dataset (which jointly optimizes the PTv3 backbone, the Pre-LN Transformer NOCS predictor, and the cross-modal pose estimator):Bashpython train.py --config config/REAL/camera_real.yaml
EvaluateTo evaluate the trained model on the REAL275 test set:Bashpython test.py --config config/REAL/camera_real.yaml --test_epoch 30
ResultsOur method achieves state-of-the-art performance on the NOCS benchmarks. You can download our training logs and pre-trained checkpoints here (Note: insert your log/checkpoint download link here).REAL275 Testset:MethodIoU50IoU755°2cm5°5cm10°2cm10°5cmOurs (PT-6DPose)83.978.449.557.969.881.4CAMERA25 Testset:MethodIoU50IoU755°2cm5°5cm10°2cm10°5cmOurs (PT-6DPose)91.387.965.371.879.087.7VisualizationTo visualize the 3D bounding box predictions and NOCS error heatmaps, run:Bashpython visualize.py --config config/REAL/camera_real.yaml --test_epoch 30
Recommended ReferencesIf you are interested in our work, we highly recommend reading the following related state-of-the-art studies in 6D pose estimation published in The Visual Computer:CMT-6D: a lightweight iterative 6DoF pose estimation network based on cross-modal Transformer. The Visual Computer, 2025, 41(3): 2011-2027.Dense point-wise line voting for robust 6D Pose estimation in industrial bin-picking. The Visual Computer, 2025: 1-12.AcknowledgementsOur implementation leverages excellent open-source code from the following projects:NOCSPointcept (PTv3)DualPoseNetWe appreciate their generous contributions to the 3D vision community.LicenseOur code is released under the MIT License (see LICENSE file for details).ContactFor any questions regarding the code or paper, please contact:Baiyang Wang: 202320482@mail.sdu.edu.cnHua Li (Corresponding Author): sdlh@sdu.edu.cn
