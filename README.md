# Enhancing Industrial Robotic Grasping through Point Transformer-Based Category-Level 6D Pose Estimation

**PT-6DPose** integrates Point Transformer, Transformer-based NOCS prediction, and cross-modal attention for robust RGB-D fusion, achieving state-of-the-art performance on NOCS-REAL275.

> **📢 Important Note:** > This repository contains the official code and dataset instructions for the manuscript currently submitted and under review at **The Visual Computer**. If you find this codebase, our pre-trained models, or our methodology helpful in your research, we strongly encourage and kindly request that you cite our manuscript (see the Citation section below). 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx) *(Note: Please replace with your actual Zenodo/GitHub DOI once generated to ensure long-term accessibility)*

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
