# Enhancing Industrial Robotic Grasping through Point Transformer-Based Category-Level 6D Pose Estimation

**PT-6DPose** integrates Point Transformer, Transformer-based NOCS prediction, and cross-modal attention for robust RGB-D fusion, achieving state-of-the-art performance on NOCS-REAL275.

> **📢 Important Note:**  
> This repository contains the official code and dataset instructions for the manuscript currently submitted and under review at **The Visual Computer**.  
> If you find this codebase, our pre-trained models, or our methodology helpful in your research, we strongly encourage and kindly request that you cite our manuscript (see the Citation section below).
>


---

## Abstract

In the realm of Industry 4.0 and smart manufacturing, 6D object pose estimation is pivotal for automating tasks such as industrial bin-picking and robotic assembly. Addressing the limitations of existing methods in handling unstructured environments, this paper introduces a novel framework for category-level 6D object pose estimation using Point Transformer. Our approach leverages multi-scale self-attention mechanisms for robust point cloud feature extraction and incorporates a Transformer-based NOCS predictor for stable coordinate prediction. A cross-modal attention mechanism is integrated to fuse RGB and depth features effectively. Experiments on the NOCS-REAL275 dataset demonstrate our method's superiority, achieving an IoU50 of 83.9% and a 5°2cm pose accuracy of 49.5%, outperforming state-of-the-art methods. This framework shows significant potential for real-world industrial robotic applications.

---

## Environment Settings

The code has been tested with:

- Python 3.9  
- PyTorch 1.12  
- CUDA 11.3  

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
pip install opencv-python
```

Since our backbone relies on Point Transformer V3, please configure the Pointcept library:

```bash
cd model/pointcept
python setup.py install
```

---

## Data Processing

### NOCS Dataset (CAMERA25 & REAL275)

Our framework relies on the standard NOCS benchmarks.

1. Download and preprocess the dataset following the standard NOCS format.  
2. Download the instance masks and segmentation results.  
3. Organize your data directory as follows:

```plaintext
data
├── CAMERA25
│   ├── train
│   └── val
├── REAL275
│   ├── train
│   └── test
├── obj_models
└── segmentation_results
```

---

## Training

To train the model on the NOCS dataset (jointly optimizing the PTv3 backbone, the Pre-LN Transformer NOCS predictor, and the cross-modal pose estimator):

```bash
python train.py --config config/REAL/camera_real.yaml
```

---

## Evaluation

To evaluate the trained model on the REAL275 test set:

```bash
python test.py --config config/REAL/camera_real.yaml --test_epoch 30
```

---

## Results

Our method achieves state-of-the-art performance on the NOCS benchmarks.  
(Insert your training logs and pre-trained checkpoint download links here.)

### REAL275 Test Set

| Method            | IoU50 | IoU75 | 5°2cm | 5°5cm | 10°2cm | 10°5cm |
|-------------------|-------|-------|-------|-------|--------|--------|
| Ours (PT-6DPose)  | 83.9  | 78.4  | 49.5  | 57.9  | 69.8   | 81.4   |

### CAMERA25 Test Set

| Method            | IoU50 | IoU75 | 5°2cm | 5°5cm | 10°2cm | 10°5cm |
|-------------------|-------|-------|-------|-------|--------|--------|
| Ours (PT-6DPose)  | 91.3  | 87.9  | 65.3  | 71.8  | 79.0   | 87.7   |

---

## Visualization

To visualize 3D bounding box predictions and NOCS error heatmaps:

```bash
python visualize.py --config config/REAL/camera_real.yaml --test_epoch 30
```

---

## Recommended References

If you are interested in 6D pose estimation in industrial settings, we recommend:

- *CMT-6D: A Lightweight Iterative 6DoF Pose Estimation Network Based on Cross-Modal Transformer*. The Visual Computer, 2025.  
- *Dense Point-Wise Line Voting for Robust 6D Pose Estimation in Industrial Bin-Picking*. The Visual Computer, 2025.

---

## Acknowledgements

Our implementation builds upon several excellent open-source projects:

- NOCS  
- PointTransformerV3


We sincerely appreciate their valuable contributions to the 3D vision community.

---

## Citation

If you use our code or models in your research, please cite:

```bibtex
@article{wang2026enhancing,
  title={Enhancing Industrial Robotic Grasping through Point Transformer-Based Category-Level 6D Pose Estimation},
  author={Wang, Baiyang and Wang, Xujian and Fang, Ming and Wang, Hongjun and Li, Hua},
  journal={The Visual Computer},
  note={Under Review},
  year={2026}
}
```

---

## License

This project is released under the MIT License (see the LICENSE file for details).

---

## Contact

- **Baiyang Wang**: 202320482@mail.sdu.edu.cn  

