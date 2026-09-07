# Temporal Representation Learning of Teaching Styles and Its Impact on Lecturer Performance

Official implementation of the paper: **Temporal Representation Learning of Teaching Styles and Its Impact on Lecturer Performance**.

This repository contains the source code for LMS log preprocessing, sequence modeling using LSTM and Transformer Autoencoder architectures, baseline models, and pedagogical style extraction.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

## Overview
Traditional LMS evaluations often rely on aggregate metric calculations. This repository implements a framework for sequence modeling to capture temporal dependencies and pedagogical chronotaxis, outperforming static aggregate models.

## Requirements
Ensure you have Python 3.8+ installed. Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset
Due to privacy policies (GDPR/PDP), the full raw dataset containing 144,178 LMS logs cannot be made public. A small, anonymized subset is provided in `sample_data.csv` for reproducibility.

## Usage
1. **Preprocessing**: 
   ```bash
   python preprocessing.py
   ```
2. **Train Baselines**:
   ```bash
   python baselines.py
   ```
3. **Train Sequential Models**:
   ```bash
   python train.py
   ```
4. **Evaluate**:
   ```bash
   python evaluate.py
   ```
5. **Visualize**:
   ```bash
   python visualize.py
   ```

## Repository Structure
- `preprocessing.py`: Converts raw LMS logs to chronological sequential tokens.
- `models.py`: PyTorch implementations of LSTM and Transformer Autoencoder.
- `baselines.py`: Implementation of static aggregate features for Decision Tree, Random Forest, SVR.
- `train.py`: Training loops for sequential models.
- `evaluate.py`: Calculates RMSE, MAE, R^2.
- `visualize.py`: Generates t-SNE latent projection and scatter plots.
- `sample_data.csv`: Anonymized dummy sample data.

## Citation
If you find this code useful for your research, please cite our paper:
```bibtex
@article{sufyaldy2026trl,
  title={Temporal Representation Learning of Teaching Styles and Its Impact on Lecturer Performance},
  author={Sufyaldy and Kudin, M. and Ahsan, M. and Maemunah and Kavari, K. M.},
  journal={Journal of Vocational, Informatics and Computer Education},
  volume={1},
  number={1},
  year={2026},
  doi={10.66053/voice.xxx.xxx}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
