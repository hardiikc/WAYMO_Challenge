# Trajectory Prediction with GRU & LSTM on the Waymo Open Dataset

## Overview
This repository contains the implementation of **GRU (Gated Recurrent Units)** and **LSTM (Long Short-Term Memory)** models for **trajectory prediction** in autonomous driving, as part of the **Waymo Open Dataset Challenge 2024 – Sim Agents**.

The project explores:
- Efficient **preprocessing** of large-scale sequential datasets (TFRecords → NPZ format)  
- **Sequence-to-Sequence architectures** (Encoder-Decoder) for predicting 8s of future trajectories from 1s of historical data  
- Comparative performance of **GRU vs. LSTM** in terms of convergence speed, accuracy, and computational efficiency  

---

## Repository Structure

1. **Documentation**
   - `/docs` : Problem statement, methodology details, workflow diagrams, and evaluation results  

2. **Source Code**
   - `/src` :
     - `preprocessing.py` : TFRecord → NPZ conversion, trajectory cleaning  
     - `lstm_model.py` : LSTM Seq2Seq model  
     - `gru_model.py` : GRU Seq2Seq model  
     - `train.py` : Training loops with Adam optimizer + learning rate scheduler  
     - `evaluate.py` : Metrics (MAE, RMSE) and error visualization  
     - `visualize.py` : Plot ground-truth vs predicted trajectories  

3. **Data**
   - `/data` : Processed dataset samples from **Waymo Open Motion Dataset (WOMD)**  
     - `raw/` : Original TFRecords  
     - `processed/` : NPZ files for training/testing  

4. **Results**
   - `/results` : Prediction graphs, training loss curves, evaluation metrics  

5. **Models**
   - `/models` : Trained checkpoints for LSTM and GRU  

---

## Setup & Installation

### Prerequisites
- Python 3.8+  
- Install dependencies:
```bash
pip install -r requirements.txt
