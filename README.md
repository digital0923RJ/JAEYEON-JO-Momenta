# JAEYEON-JO | Momenta Audio Deepfake Detection

> **Real-Time Audio Deepfake Detection using LCNN with Max-Feature-Map (MFM)**

---

## Project Overview

This project aims at detecting audio deepfakes in **real-time**, leveraging a lightweight **LCNN (Lightweight Convolutional Neural Network)** enhanced by **Max-Feature-Map (MFM)** layers. While models like ResNet2 and Wav2Vec2.0 offer higher accuracy, LCNN+MFM provides an optimal balance between performance and real-time capability, making it ideal for practical scenarios.

---

## Model Comparison

| Model         | Key Innovations                                                 | Performance                       | Real-Time Suitability | Context Handling |
|---------------|-----------------------------------------------------------------|-----------------------------------|-----------------------|------------------|
| **ResNet2**   | Residual Learning, Dilated Convolutions, Attention Pooling      | EER ≤ **1.6%**, Accuracy **98%+** | ⚠️ Moderate           | ✅ Excellent     |
| **LCNN + MFM**| Lightweight CNN, Max-Feature-Map (MFM)                          | EER ~ **2.6–3%**, Accuracy ~95%   | ✅ High               | ⚠️ Limited       |
| **Wav2Vec2.0**| Transformer-based self-supervised architecture                  | EER ≤ **2.2%**, Accuracy **97%+** | ❌ Low                | ✅ Best          |

---

## Why LCNN with MFM?

- **Real-time detection capability**
- **Optimized for edge devices**
- **Enhanced noise suppression via MFM**
- **Balanced trade-off between speed and accuracy**

---

## Implementation Pipeline

| Component          | Description                                           |
|--------------------|-------------------------------------------------------|
| **Preprocessing**  | Log-Mel spectrograms, SpecAverage augmentation        |
| **Data Caching**   | Cached spectrograms (.npy files)                      |
| **Data Splitting** | Stratified train/test splits                          |
| **Model Structure**| LCNN with MFM layers replacing standard ReLU          |
| **Training/Eval**  | BCELoss, Adam optimizer; AUC, ACC, EER metrics        |
| **Visualization**  | Performance charts (`lcnn_performance.png`)           |

---

## Key Challenges & Solutions

| Category                | Challenge                                   | Solution                                        |
|-------------------------|---------------------------------------------|-------------------------------------------------|
| **Dataset Handling**    | Large dataset slowing training               | Cached spectrogram preprocessing results        |
| **File Compatibility**  | Inconsistent file extensions (.wav vs .WAV)  | Case-insensitive audio file loading             |
| **Duration Bias**       | Audio length variability                     | Standardized audio clips to 5 seconds           |
| **Overfitting Risk**    | Limited dataset causing overfitting          | SpecAverage augmentation and MFM layers         |
| **Dataset Realism**     | Lack of genuine speech data                  | Integrated real speech (LJSpeech dataset)       |

---

## Performance Results

| Metric | LCNN + MFM Performance |
|--------|------------------------|
| AUC    | **0.8165**             |
| ACC    | **0.7363**             |
| EER    | **0.2620**             |


![image](https://github.com/user-attachments/assets/a5dd4b3d-7f99-4ea9-9c6a-641919f9711d)



---

## Strengths & Weaknesses

### ✅ Strengths
- Real-time inference suitable for live applications
- Robust to noise with MFM layer
- Lightweight and easily deployable on edge devices

### ⚠️ Weaknesses
- Limited contextual analysis without additional modules
- Slight increase in complexity over basic LCNN

---

## Future Improvements

- Incorporate diverse, noisy, multilingual, and real-world audio samples
- Test newer deepfake audio generation techniques
- Experiment with advanced augmentations
- Evaluate ensemble methods combining LCNN with ResNet or distilled Wav2Vec models

---

## Deployment Strategy

### Data Expansion
- Collect varied audio from realistic scenarios

### Cloud Automation
- Use AWS SageMaker/Azure ML for automated training pipelines
- Manage datasets efficiently using cloud storage solutions

### Real-Time Inference
- Implement sliding window methods for continuous audio analysis

### Edge Deployment
- Export models via ONNX, TorchScript, or TFLite for mobile 

### Monitoring & Maintenance
- Develop dashboards for real-time model performance tracking
- Automate regular model updates with new data

---

## Final Thoughts

Integrating MFM into LCNN significantly improved noise robustness and inference efficiency, providing a reliable real-time solution for detecting audio deepfakes. Continuous adaptation and data enrichment are essential to maintain effectiveness as deepfake technologies evolve.

---
