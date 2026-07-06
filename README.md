# 🌿 Monitoring *Spartina alterniflora* Removal Dynamics Across Coastal China Using Time Series Sentinel-1 Imagery

Official implementation of the **Remote Sensing of Environment (RSE, 2025)** paper for monitoring the spatial extent and removal timing of *Spartina alterniflora* across coastal China using dense Sentinel-1 SAR time series.

[![Paper](https://img.shields.io/badge/Paper-RSE%202025-orange)](https://www.sciencedirect.com/science/article/abs/pii/S0034425725002172)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Google Earth Engine](https://img.shields.io/badge/Platform-Google%20Earth%20Engine-green)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

# 📖 Overview

*Spartina alterniflora* is one of the most aggressive invasive plant species in coastal wetlands, posing significant threats to native ecosystems and biodiversity. Large-scale ecological restoration projects have been implemented across China to remove *S. alterniflora*. However, monitoring the spatial extent and timing of removal activities over broad geographic regions remains challenging.

This repository provides the official implementation of the **STS-CTD** framework, which utilizes dense Sentinel-1 SAR time series to automatically identify *S. alterniflora* removal areas and estimate their removal timing across the entire coastline of China.

---

# ✨ Highlights

- 🌿 Developed the **STS-CTD** framework for nationwide monitoring of *Spartina alterniflora* removal.

- 🛰️ Utilized dense Sentinel-1 SAR time series to capture temporal changes associated with vegetation removal.

- 📅 Simultaneously mapped both **removal extent** and **removal timing** across China's coastal wetlands.

- 🇨🇳 Produced the first nationwide high-resolution dataset of *S. alterniflora* removal dynamics.

---

# 🚀 Quick Start

The trained model is provided in the `model_data` directory.

To visualize the prediction results:

```bash
python Run_result.py
```

> **Note**
>
> The released model parameters are trained for the study area described in the paper and may not generalize directly to other regions.

---

# 🛰️ STS-CTD Framework

<p align="center">
<img src="https://github.com/user-attachments/assets/4b8a25c7-a9d8-41bf-b953-3ee84fed28af" width="1000">
</p>

**Figure 1.** Overall workflow of the proposed **Spartina Time-Series Change Tracking and Detection (STS-CTD)** framework for monitoring *S. alterniflora* removal using Sentinel-1 SAR imagery.

---

# 🇨🇳 National Mapping Results

<p align="center">
<img src="https://github.com/user-attachments/assets/ca8dac7e-909e-446f-8cc5-fb232b0bfccb" width="1000">
</p>

**Figure 2.** National maps of *Spartina alterniflora* removal extent and removal timing across coastal China. Insets illustrate representative estuaries and coastal wetlands, including the Yellow River Estuary, Yancheng, Chongming Island, Hangzhou Bay, Sanmen Bay, Leqing Bay, Sansha Bay, Dingzi Port, and Zhangjiang Estuary.

---

# 🔍 Local Examples

<p align="center">
<img src="https://github.com/user-attachments/assets/710296db-6746-46de-9a07-55b3126ad92a" width="1000">
</p>

**Figure 3.** Representative local examples showing mapped removal extent and removal timing together with corresponding Sentinel-1 false-color composites (σVH, σVV, and CR), demonstrating the capability of the proposed method to capture removal dynamics in different coastal environments.

---

# 📂 Repository Structure

```text
.
├── model_data/          Trained STS-CTD model
├── data/                Sample data
├── Image/               Figures
├── Run_result.py        Inference script
├── requirements.txt
└── README.md
```

---

# 📊 Data

This project is developed using:

- Sentinel-1 Ground Range Detected (GRD) imagery
- Google Earth Engine
- Python

---

# 📄 Publication

Min, Y., Ke, Y., Zhuo, Z., Qi, W., Li, J., Li, P., & Zhao, N. (2025).

**Monitoring *Spartina alterniflora* removal dynamics across coastal China using time series Sentinel-1 imagery.**

*Remote Sensing of Environment*, **326**, 114813.

DOI:
https://doi.org/10.1016/j.rse.2025.114813

---

# 📖 Citation

If you find this repository useful, please cite:

```bibtex
@article{Min2025STSCTD,
  title={Monitoring Spartina alterniflora removal dynamics across coastal China using time series Sentinel-1 imagery},
  author={Min, Yukui and Ke, Yinghai and Zhuo, Zhaojun and Qi, Weichun and Li, Jinyuan and Li, Peng and Zhao, Nana},
  journal={Remote Sensing of Environment},
  volume={326},
  pages={114813},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.rse.2025.114813}
}
```

---

# 🙏 Acknowledgements

This work is based on the following open datasets and platforms:

- Sentinel-1 Mission (ESA)
- Google Earth Engine

---

# 📜 License

This project is released under the MIT License.
