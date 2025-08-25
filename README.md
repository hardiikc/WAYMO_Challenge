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

pip install -r requirements.txt
  
---

## 📚 Main Libraries
- **PyTorch**  
- **NumPy / Pandas**  
- **Matplotlib / Seaborn**  
- **TensorFlow** (for preprocessing)  

---

##  Workflow

### 1. Data Preprocessing
- Converted **TFRecords → NPZ** for efficient training  
- Normalized sequential data (positions, velocities, yaw)  
- Generated training samples: **11 timesteps history → 80 timesteps prediction**  

### 2. LSTM Model
- Seq2Seq Encoder-Decoder with **2 hidden layers (128 neurons each)**  
- **Dropout 0.3** to prevent overfitting  
- Loss: **Mean Squared Error (MSE)**  
- Optimizer: **Adam + learning rate scheduling**  

### 3. GRU Model
- Similar Seq2Seq setup with **GRU cells (2 layers, 64 neurons)**  
- Simplified gating mechanism → faster convergence  
- Loss: **MSE**, Optimizer: **Adam**  

### 4. Training
- Mini-batch training with autoregressive decoding  
- Debugging assertions for tensor shape validation  
- Adjusted batch size due to GPU memory constraints  

### 5. Evaluation
- Metrics: **MAE (Mean Absolute Error), RMSE (Root Mean Squared Error)**  
- Visual comparison of predicted vs ground truth trajectories  

---

##  Results

### LSTM
- Learned temporal dependencies effectively  
- Produced smooth trajectory predictions  
- Slower convergence  

### GRU
- Faster training convergence  
- Comparable accuracy with fewer parameters  
- Computationally more efficient  

### Key Findings
- GRU outperformed LSTM in terms of **training speed** while achieving **similar prediction accuracy**  
- Visualizations showed realistic predicted paths aligned with ground-truth data  
- Final model errors:  
  - **Avg Error ≈ 1.25 m**  
  - **Final Position Error ≈ 2.80 m**  

---

##  Highlights
- End-to-end pipeline: **TFRecords → Preprocessing → Seq2Seq → Evaluation**  
- Modular design: swap easily between **GRU, LSTM, CNN, or Transformer models**  
- Detailed **visualizations** for debugging and interpretability  

---

##  Future Work
- Integrate **Transformer-based motion forecasting models** (e.g., BERT4Motion, SceneDiffuser)  
- Multi-agent coordination and interaction-aware prediction  
- Real-time inference optimization for **autonomous vehicle deployment**  
- Hybrid approaches combining **Reinforcement Learning + Seq2Seq prediction**

```bash
