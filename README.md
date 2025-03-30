# PGD Targeted Attack on MNIST (Success Rate Only)

This project performs **PGD (Projected Gradient Descent)** attacks on a CNN trained with the **MNIST dataset**, focusing on **targeted attacks**.

## Objective

- Train a simple CNN on MNIST
- Perform **targeted PGD attacks** for each target class (0~9)
- Print the attack **success rate (%)** for each target class
- No visualization, only numerical results

## Attack Details

- Attack type: **Targeted PGD**
- Dataset: **MNIST**
- Model: Simple CNN with two convolutional layers
- PGD Parameters:
  - `eps = 0.3`
  - `eps_step = 0.03`
  - `k = 10` iterations

## How to Run

```bash
pip install -r requirements.txt
python test.py
