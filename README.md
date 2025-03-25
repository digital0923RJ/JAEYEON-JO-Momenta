# JAEYEON-JO - Momenta Audio Deepfake Detection

## Summary

To briefly summarize, I selected three models—ResNet2, LCNN, and Wav2Vec2.0—for comparison.  
After reviewing several papers, I found that most models struggle to generalize well.  
Rather than prioritizing maximum accuracy, I focused on **real-time detection** of audio deepfakes.  
If this were a competition or a task purely about performance, I would have chosen heavier models.  
However, if a model can reliably achieve ~90% accuracy, my priority is to **respond quickly** with a lightweight model in real-world scenarios.

---

## Audio Deepfake Detection Models Comparison Table

| Model         | Key Technical Innovation                                       | Reported Performance                       | Why It's Promising                                                                 | Potential Limitations                                                           |
|---------------|---------------------------------------------------------------|---------------------------------------------|--------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **ResNet2**   | - Residual learning<br>- Dilated convolutions & attention pooling | - EER ≤ **1.6%** (ADD 2022)<br>- Accuracy **98%+** | - Powerful deep feature extraction<br>- Robust across conditions<br>- Great for conversational detection | - High computational cost<br>- Needs Transformer for long-range context         |
| **LCNN**      | - Lightweight CNN<br>- Max-Feature-Map (MFM) for noise suppression | - EER ~**3–5%**<br>- Real-time capable       | - Ideal for **real-time detection**<br>- Edge deployable<br>- Noise-tolerant         | - Limited context modeling<br>- Weaker for complex, long speech segments        |
| **Wav2Vec2.0**| - Self-supervised Transformer<br>- Raw audio input             | - EER ≤ **2.2%**<br>- Accuracy **97%+**     | - Captures long context, emotion, and semantics<br>- Strong few-shot performance     | - Large model size<br>- High compute requirement<br>- Hard to use on edge       |

---

### ✅ Why I Chose LCNN

As shown in the table below, LCNN is:
- ✅ Real-time capable  
- ✅ Edge-device friendly  
- ✅ Compatible with log-Mel features  
- ✅ Performs well with lightweight setups

| Model         | Accuracy | EER     | Real-time Capability       | Conversational Analysis     |
|---------------|----------|---------|----------------------------|-----------------------------|
| **ResNet2**   | 98%+     | ≤1.6%   | ⚠️ Moderate (needs optimization) | ✅ Highly suitable           |
| **LCNN**      | ~95%     | 3–5%    | ✅ Very high                | ⚠️ Limited                   |
| **Wav2Vec2.0**| 97%+     | ≤2.2%   | ❌ Not suitable             | ✅ Best performance          |

While LCNN may have slightly lower accuracy, it’s the most practical for real-time use.  
Because LCNN lacks deep architecture, I focused on **strong preprocessing** to extract meaningful features.  
I chose `log-Mel + SpecAverage` as my primary preprocessing method, commonly used in research.  
SpecAverage helps preserve feature diversity by masking values with average energy.  
Mel spectrograms are sensitive to duration, so I normalized all inputs to **5 seconds** using padding.

---

## Part 3: Documentation & Analysis

### 1️⃣ Implementation Challenges

| **Challenge**                        | **Details**                                                                                                      |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Large Dataset Handling              | Cached extracted log-Mel spectrograms as `.npy` files to avoid recomputation and speed up training.             |
| File Format Inconsistencies         | Used case-insensitive glob pattern `**/*.[wW][aA][vV]` to handle both `.wav` and `.WAV` files uniformly.        |
| Overfitting Risk & Duration Bias    | Trimmed/padded all audio clips to 5 seconds to remove duration-based bias.                                      |
| Model Generalization                | Used SpecAverage augmentation to help generalize to unseen fake audio.                                          |
| Lack of Real Human Speech Data      | Added LJSpeech dataset to represent genuine human voice and increase dataset realism.                           |

### 2️⃣ Solutions to Challenges

| **Challenge**             | **Solution**                                                                 |
|---------------------------|------------------------------------------------------------------------------|
| Large dataset             | Cached `.npy` log-Mel features for reuse                                     |
| File format issues        | Used `**/*.[wW][aA][vV]` glob pattern                                         |
| Audio length bias         | Fixed all audio to 5 seconds via padding/trimming                            |
| Overfitting               | Applied SpecAverage and selected a lightweight LCNN architecture             |
| Lack of real speech data  | Included LJSpeech dataset for authentic audio                                |

### 3️⃣ Assumptions Made

- **5 seconds** of audio contains sufficient signal for detecting deepfake cues.
- **log-Mel + SpecAverage** preprocessing captures richer features than MFCC alone.
- **LCNN** is sufficient for baseline detection in real-time applications.
- Case differences in file extensions (e.g., `.wav` vs `.WAV`) are syntactic, not acoustic.

---

### 📊 Model Pipeline Summary

| Component          | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| **Preprocessing**  | Applied `log-Mel + SpecAverage` using `librosa`.                            |
| **Caching**        | Saved spectrograms as `.npy` for faster future access.                      |
| **Data Split**     | Used stratified `train_test_split` to balance labels.                       |
| **Model**          | LCNN with ReLU activations, 2 CNN layers, and AdaptiveAvgPooling.           |
| **Training/Eval**  | Used `BCELoss` and `Adam`; tracked AUC, ACC, and EER per epoch.             |
| **Visualization**  | Saved performance bar chart as `lcnn_performance.png`.                      |
| **Output**         | Printed metrics to console; exported visualization image.                   |

---

### ✅ High-Level Model Description

- **Input:** Each 5-second audio clip is transformed into a log-Mel spectrogram with SpecAverage masking.
- **Architecture:** LCNN with 2 convolutional layers + ReLU + BatchNorm + AdaptiveAvgPool, followed by a fully connected layer with Sigmoid.
- **Loss:** Binary Cross-Entropy (BCELoss)
- **Metrics:**  
  - AUC: Area under the ROC curve  
  - ACC: Classification accuracy  
  - EER: Equal Error Rate — when False Positive Rate equals False Negative Rate

---

### ✅ Performance Results

- **LCNN Epoch 20:**  
  `Loss = 0.6353`  
  `AUC = 0.7233`  
  `ACC = 0.5421`  
  `EER = 0.3371`

---

### 🔍 Strengths & Weaknesses

#### ✅ Strengths

- **Lightweight & Fast:** Suitable for edge deployment with limited resources.  
- **Effective on Short Clips:** Handles 5-second inputs well.  
- **Interpretable:** Simple architecture makes debugging and visualization easier.

#### ⚠️ Weaknesses

- **Overfitting Risk:** Tends to memorize small datasets.  
- **Limited Context:** LCNN lacks long-range modeling like Transformers.  
- **Performance Ceiling:** Struggles to surpass 75–80% AUC without richer features or ensembles.

---

### ✅ Suggestions for Future Improvements

- Use more realistic datasets: **ASVspoof**, **FakeAVCeleb**, call center audio.
- Add data from **noisy, multilingual, and modern TTS** environments (e.g., VITS, Bark, Vall-E).
- Incorporate advanced augmentation: SpecMix, pitch shifting, time-stretching, background noise.

---

## 3️⃣ Reflection

### Q1 & Q2: Major Challenges & Real-World Performance

| Category                | Challenge / Consideration                                                       | Solution / Implementation                                                  |
|-------------------------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| File Format Issues   | Case mismatch in `.wav` vs `.WAV` created loading bugs                          | Case-insensitive loader and file handling                                   |
| Preprocessing Bias   | Model overfitting to audio duration rather than content                         | Applied fixed-length 5s input + SpecAverage augmentation                    |
| Model Selection       | Needed a balance of speed vs performance for production                         | Chose LCNN for fast, efficient deployment                                   |
| Realism in Dataset   | Many datasets are synthetic only                                                | Included real human speech via LJSpeech dataset                             |

---

### Q3: What Additional Data Would Improve Performance?

- **Noisy real-world audio** (e.g., phone calls, YouTube, podcasts)  
- **Modern fake audio** (e.g., Vall-E, Bark, VITS samples)  
- **Multilingual datasets** to improve generalization  
- **Augmentations** like SpecMix, pitch shifting, time-stretching, noise injection

---

### Q4: Deployment Strategy

#### Dataset Expansion
- Add data from diverse sources, languages, and TTS models for generalization.

####  Cloud-Based Automation
- Use **AWS SageMaker** or **Azure ML Pipelines** to automate training and deployment workflows.
- Store and manage datasets in **S3** or **Azure Blob Storage**.

#### Inference Pipeline
- Apply 5-second **sliding window** over long audio for real-time chunk-wise prediction.
- Aggregate outputs for final decisions.

#### Edge Deployment
- Deploy **LCNN** on mobile or embedded devices for fast inference:
  - Mobile phones (iOS/Android)
  - Raspberry Pi / Jetson / IoT devices
- Convert model to **ONNX**, **TorchScript**, or **TFLite**.

#### Monitoring & Retraining
- Set up real-time dashboards (CloudWatch, Grafana)  
- Monitor prediction confidence and drift  
- Retrain on new samples as synthesis methods evolve




# JAEYEON-JO-Momenta

간단하게 우선 설명을 하자만, 나는 ResNet2, LCNN, Wav2Vec2.0를 선정하였다. 
논문을 읽어보니, 대부분의 모델이 아직까지는 일반화 시 성능이 떨어지는 것을 보고 성능의 우선보다는  real-time에서 deepfake를 실시간 감지하는 것에 최우선을 두었다. 만약 이것이 대회거나 성능 향상에만 초점을 맞췄으면 computation이 많이 요구되는 모델을 사용했을 것인데, 그게 아니라, 정확도가 약 90%까지 확보됬다고 생각하면 그냥 실시간 정보에 빠르게 대응할 수 있는 모델을 선정했다. 

## 🔍 Audio Deepfake Detection Models Comparison Table 아래는 논문을 읽고 내가 3개 선택한 모델 분석이다. 

| Model       | Key Technical Innovation | Reported Performance | Why It's Promising | Potential Limitations |
|-------------|---------------------------|-----------------------|----------------------|------------------------|
| **ResNet2** | - Residual learning structure<br>- Can incorporate dilated convolution & attention pooling | - EER ≤ **1.6%** (ADD 2022)<br>- Accuracy **98%+** | - Strong deep feature extraction<br>- Robust across varied conditions<br>- Well-suited for conversation-based detection | - High computational cost<br>- Limited ability to capture long-term context (needs Transformer integration) |
| **LCNN**    | - Lightweight CNN with Max-Feature-Map (MFM)<br>- Effective noise suppression | - EER around **3–5%**<br>- Real-time capable | - Excellent for **real-time detection**<br>- Suitable for edge devices<br>- Performs well in noisy environments | - Lacks deep contextual understanding when used alone<br>- Not ideal for complex conversations |
| **Wav2Vec2.0** | - Self-supervised learning<br>- Raw audio input + Transformer-based architecture | - EER ≤ **2.2%**<br>- Accuracy **97%+** | - Outstanding at capturing long context, emotion, and meaning<br>- Performs well with **limited labeled data**<br>- Ideal for real conversation analysis | - Large model size and high compute cost<br>- Requires optimization for real-time deployment |

---
### 🔍 그리고 나는 이 중에서 LCNN을 선택했다. Why my choice is LCNN?
아래 표에서 보이듯, 
- ✅ Real-time capable  
- ✅ Easy to deploy on edge devices  
- ✅ Compatible with log-Mel features  
- ✅ Performs well with low-compute environments

| Model       | Accuracy | EER     | Real-time Capability | Conversational Analysis |
|-------------|----------|---------|------------------------|--------------------------|
| ResNet2     | 98%+     | ≤1.6%   | ⚠️ Moderate (needs optimization) | ✅ Highly suitable |
| LCNN        | ~95%     | 3–5%    | ✅ Very high           | ⚠️ Limited |
| Wav2Vec2.0  | 97%+     | ≤2.2%   | ❌ Needs high resources | ✅ Best performance |



보면 LCNN은 정확도가 95%까지 올라가고 Very high real-time capacity가 있다고 생각되어 LCNN 모델을 선택했다. 
대신, 전처리가 중요하여 전처리를 잘하려고 노력했다.
대략적으로 크게 전처리하는 방식은 MFCC, Mel, SpecAverage가 논문보니까 많이 쓰였더라. 그래서 나도 이 방법을 통해 전처리 하려고 했다. 
LCNN은 정확도가 상대적으로 낮은 모델이라 전처리를 통해 최대한 많은 feature 특징들을 뽑으려고 했다. 
그래서 1번의 전처리본다는 log-Mel + SpecAvaerge 전처리를 선택했다. SpecAverage를 선택한 것은 마스킹을 평균으로 조절하면서 특성을 잘 뽑을 수 있기에 선택했다. 
대신 Mel은 오디오 길이에 민감하기에 모든 오디오를 5초로 통일하려고 했고 padding함 


## Part 3: Documentation & Analysis

1. Document your implementation process, including:
    - Any challenges encountered

| **Challenge**                       | **Details**                                                                                                       |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| **Large Dataset Handling**         | Cached extracted log-Mel spectrograms as `.npy` files to avoid recomputation and reduce training time.           |
| **File Format Inconsistencies**    | Used case-insensitive glob pattern `**/*.[wW][aA][vV]` to handle `.wav` and `.WAV` files uniformly.               |
| **Overfitting Risk & Duration Bias** | Applied fixed-length preprocessing: all audio clips were trimmed or zero-padded to 5 seconds to remove bias.     |
| **Model Generalization**           | Used SpecAverage augmentation to reduce overfitting and promote robustness in learning.                          |
| **Lack of Real Human Speech Data** | Added the LJSpeech dataset to introduce authentic human voice samples and improve realism.                       |
      
    - How you addressed these challenges
 | **Challenge**            | **Solution**                                                                 |
|--------------------------|------------------------------------------------------------------------------|
| Large dataset            | Used **caching (`.npy`)** of extracted log-Mel features to avoid recomputation |
| File format issues       | Implemented case-insensitive glob pattern: `**/*.[wW][aA][vV]`                |
| Audio length bias        | All audio was **padded or trimmed to 5 seconds** using fixed-length preprocessing |
| Overfitting              | Applied **SpecAverage augmentation** and used a **lightweight CNN model (LCNN)** |
| Real speech data scarcity | Added **LJSpeech dataset** to represent real human voice                    |
   

    - Assumptions made
    - **5 seconds of audio** is enough to capture the key acoustic features necessary to distinguish between real and fake speech.
- Using **log-Mel spectrograms combined with SpecAverage masking** improves generalization and reduces overfitting compared to MFCCs alone.
- A **lightweight architecture (LCNN)** is sufficient for baseline detection without requiring large-scale Transformer-based models like Wav2Vec 2.0.
- Both `.wav` and `.WAV` formats are **acoustically identical**; the issue is purely syntactic and handled during file loading.

2. Include an analysis section that addresses:
📊 Model Analysis
| 구성 요소     | 설명 |
|----------------|------|
| **전처리**      | `log-Mel + SpecAverage` 적용. `librosa`를 이용하여 오디오를 로딩하고 스펙트로그램 변환 수행. |
| **캐싱**        | `.npy` 형태로 저장하여 재사용 가능, 빠른 데이터 로딩 가능. |
| **데이터 분리** | `train_test_split`에 기반하여 stratified split 적용, 클래스 비율 유지. |
| **모델 구조**   | LCNN 기반 구조. `ReLU` 활성화 함수 사용. 2층 CNN + `AdaptiveAvgPool`로 출력 정규화. |
| **훈련/평가 루프** | `BCELoss`와 `Adam` Optimizer 사용. 매 epoch마다 `AUC`, `ACC`, `EER` 지표 출력. |
| **시각화**     | 최종 성능을 `막대그래프`로 시각화하여 `lcnn_performance.png`로 저장. |
| **최종 출력**   | 콘솔에 평가 지표 출력 + 이미지 파일 저장 (`lcnn_performance.png`). |

### ✅ Why LCNN Was Selected

- **Real-time capability:**  
  LCNN (Lightweight Convolutional Neural Network) is fast and computationally efficient—ideal for edge or mobile deployment.

- **Proven effectiveness:**  
  LCNN has been shown to perform well in fake audio detection tasks with relatively low error rates.

- **Compatibility:**  
  Works well with log-Mel spectrograms and benefits from data augmentations like SpecAverage.

- **Interpretability:**  
  Easier to visualize and understand than Transformer-based models.

### ✅ How the model works (high-level technical explanation)

- **Input:** Each audio file is converted into a log-Mel spectrogram, augmented using SpecAverage masking, and reshaped to a fixed length (5 seconds)
- **Model Architecture:** Two convolutional layers with ReLU, followed by BatchNorm and Adaptive Average Pooling.Fully connected layer with a sigmoid activation outputs the probability of the audio being fake.
- **Loss Function:** Binary Cross-Entropy Loss (BCELoss) is used since this is a binary classification problem (real vs. fake).
- **Evaluation Metrics:** AUC (Area Under ROC Curve): How well the model separates the two classes, Accuracy: Proportion of correct predictions, EER (Equal Error Rate): The point where False Positive Rate = False Negative Rate.

### ✅ Performance results on your chosen dataset
- #### LCNN Epoch 20: Loss=0.6353, AUC=0.7233, ACC=0.5421, EER=0.3371
  
### 🔍 Strengths & Weaknesses
#### 🟢 Strengths

- **Lightweight & Fast**  
  Can be deployed in low-resource environments.

- **Effective on Short Clips**  
  Works reasonably well with 5-second samples.

- **Interpretable**  
  Simpler architecture makes debugging and tuning easier.

#### 🔴 Weaknesses

- **Overfitting**  
  Model quickly memorizes small datasets.

- **Limited Context**  
  LCNN lacks long-range temporal modeling like Transformers.

- **Performance Ceiling**  
  Struggles to exceed 75–80% AUC without ensemble or richer features.

### ✅ Suggestions for future improvements
- Improve Dataset Quality & Variety:
  Use ASVspoof, FakeAVCeleb, and call center-style datasets for better generalization.
  Add noisy environments, different languages, and newer fake voice synthesis methods (VITS, Bark, Vall-E).


## 3. Reflection questions to address:

### Q1: What were the most significant challenges? & Q2: How might this approach perform in the real world?

| Category                | Challenge / Consideration                                                     | Solution / Implementation                                                                 |
|-------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| 🔄 File Format Issues   | Large dataset size, case-sensitive `.wav` vs `.WAV` caused read errors       | Used consistent loader logic and added compatibility for both `.wav` and `.WAV` extensions  |
| ⚙️ Preprocessing Dilemma | Model overfitting, generalization problems, bias due to length differences   | Applied `log-Mel + SpecAverage`, and fixed all audio to **5 seconds** using pad/crop       |
| ⚖️ Model Selection       | Trade-off between inference speed and accuracy                               | Chose **LCNN** for fast, real-time capable, lightweight architecture                        |
| 🔍 Real Data Requirement | Needed realistic audio (not just synthetic/generated)                        | Added **LJSpeech** dataset as genuine human voice source                                    |
---

### Q3: What additional data would improve performance?

- **Noisy and real-life audio**  
  e.g., phone calls, YouTube recordings, podcast snippets.

- **Fake audio from modern generators**  
  Incorporate samples from models like **Vall-E**, **Bark**, or **VITS** to stay up to date with the latest synthesis techniques.

- **Multilingual data**  
  To ensure the model generalizes across different languages and accents.

- **Advanced data augmentation techniques**  
  Such as: SpecMix, Pitch shift, Time-stretching, Background noise injection
---

### Q4: How would you deploy this?

#### 📦 Dataset Expansion for Generalization
- **Add more diverse data** to improve model generalization:
  - Noisy real-world audio (e.g., phone calls, podcasts, YouTube)
  - Multiple languages and accents
  - Fake audio from modern TTS models (e.g., Vall-E, Bark, VITS)
- Ensure dataset includes varied recording environments, devices, and speakers.

#### 🤖 Model Training Automation (Cloud)
- Use **cloud platforms like AWS or Azure** to automate model training, evaluation, and deployment:
  - Leverage **AWS SageMaker** or **Azure ML Pipelines**

#### 🧠 Inference Pipeline
- Implement **batch inference** for stored datasets and **real-time streaming inference** for live applications.

#### 📱 Edge Deployment
- Use the **LCNN model** for **low-latency, lightweight inference** on edge devices:
  - Mobile phones (iOS/Android)
  - Embedded systems (Raspberry Pi, NVIDIA Jetson)
  - IoT devices (smart assistants, surveillance)

- Convert model to **ONNX**, **TorchScript**, or **TFLite** for optimal performance on various platforms.

#### 📊 Monitoring & Retraining
- **Continuously monitor**:
  - Prediction confidence levels
  - Accuracy drift over time
  - New types of fake audio
- Set up **automated retraining pipelines** triggered by performance degradation or new data.
- Use dashboards to visualize performance (e.g., via Amazon CloudWatch, Azure Monitor, or Grafana).
---
























## 📚 Appendix

### 📁 Dataset Info

- **Real**: LJSpeech (~13,000 clips, ~10 seconds each)  
- **Fake**: MelGAN-generated audio (~5 seconds)  
- ✅ Class-balanced cache system implemented

---

### 🧪 Training Pipeline Highlights

- `extract_logmel()` ensures uniform audio duration  
- Padding applied to ensure consistent input dimensions  
- `pad_collate_fn()` used for batch-wise padding  
- Still requires **early stopping or regularization**

---

## ✅ Summary Table

| Feature                | LCNN          | ResNet2        | Wav2Vec2.0     |
|------------------------|---------------|----------------|----------------|
| Inference Speed        | ⚡ Very fast   | ⚠️ Moderate     | 🐢 Very slow    |
| Accuracy (Expected)    | 👍 ~95%        | ✅ 98%+         | ✅ 97%+         |
| EER (Expected)         | ⚠️ 3–5%        | ✅ ≤1.6%        | ✅ ≤2.2%        |
| Suitable for Streaming | ✅ Yes         | ⚠️ With tuning  | ❌ Not ideal    |

---

## 🚀 Future Work

- ✅ Integrate **multi-model ensemble** (LCNN + ResNet2 + distilled Wav2Vec2.0)  
- ✅ Fine-tune on **ASVspoof 2021**, **FakeAVCeleb**, or more recent datasets  
- ✅ Experiment with **attention pooling**, **SpecMix**, **SpecAugment++**  
- ✅ Build a **frontend dashboard** for live detection and visualization

---





# 🎧 Audio Deepfake Detection using log-Mel + SpecAverage + LCNN

> Lightweight audio deepfake detection with strong real-time capability and generalization focus.  
> Explored multiple models and preprocessing methods.  
> Final implementation uses LCNN with SpecAverage-augmented log-Mel features.

---

## 📌 Overview

This project focuses on building an efficient and practical pipeline for detecting AI-generated human speech using lightweight CNN-based models. The goal is real-time applicability while maintaining robust detection capability.

---

## 1. ⚙️ Implementation Documentation

### ✅ Challenges Encountered

1. **Large Fake Dataset**
   - Slow preprocessing and loading due to dataset size
   - Limited generalization: all audio was generated
   - Real vs. fake audio had different durations (10s vs. 5s)

2. **Wav2Vec2.0 Inference Cost**
   - Too large for real-time or mobile deployment
   - Slower than lightweight CNNs like LCNN and ResNet2

3. **File Extension Issues**
   - `.wav` vs `.WAV` (case-sensitive) caused loading failures

4. **Overfitting Risk**
   - Small dataset + deep CNNs = higher chance of memorization

5. **Preprocessing Dilemma**
   - MFCC compresses too much
   - Log-Mel is better but vulnerable to overfitting
   - SpecAverage helps with augmentation but needs tuning

6. **Length-Based Bias**
   - Model might learn to classify based on input length (real=10s, fake=5s)

---

### 🛠️ How These Were Addressed

- ✅ Added real audio: **LJSpeech** dataset
- ✅ Switched to lightweight models: **LCNN**, **ResNet2**
- ✅ Preprocessing: used **log-Mel + SpecAverage**
- ✅ Fixed-length audio: trimmed/padded all to **5 seconds**
- ✅ Created custom `extract_logmel()` with duration constraint

```python
def extract_logmel(path, sr=16000, n_mels=40, duration=5.0):
    y, _ = librosa.load(path, sr=sr)
    target_len = int(sr * duration)
    y = np.pad(y, (0, max(0, target_len - len(y))))[:target_len]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel)
    return torch.tensor(spec_average_augment(logmel)).float().unsqueeze(0)


# 🧾 Part 3: Documentation & Analysis

## 1. Implementation Documentation

### ✅ Challenges Encountered

1. **데이터셋의 크기 문제**  
   다운로드한 fake 오디오 데이터셋이 너무 커서 전처리와 로딩에 오랜 시간이 걸렸고, 모두 generated audio라서 일반화에 제한이 있을 수 있었습니다.
   실제로 모델을 돌려볼 시간이 많이 없었음. 

2. **Wav2Vec 2.0 모델의 실시간 처리 한계**  
   Wav2Vec2.0은 powerful하긴 하지만, 크기가 커서 real-time 적용에는 부적합했습니다. 특히 모바일/경량화가 중요한 환경에서는 사용이 어려움.

3. 데이터 전처리: 과제를 위한 논문들에서 대부분의 audio deepfake detection model의 한계점을 일반화라고 함. 즉,
   데이터 전처리를 어떻게 해야 일반화 시 떨어지는 성능을 최대한 방지할 수 있을까 고민함. Noise가 없는 고성능 파일말고 noise가 있는 파일, 그리고 또 최신의 기술로 생성된 deepfake audio 등 다양한 데이터셋 확보가 필요함 
   MFCC:
   Mel:
   SpecAverage:  위의 3가지 전처리 방법으로는 모든 deepfake audio detection에 한계가 있음. 
 
4. 일반 머신러닝 모델 사용하지 않기. 
   일반 머신러닝 모델을 사용하니까 활성화 함수가 gradient를 0으로 만듬. 즉, 깊이 있는 신경망을 건축할 수가 없음. 
---

### ✅ How Challenges Were Addressed

1. **실제(real) 오디오 데이터셋 추가 확보**  
   LJSpeech 데이터셋(`"/content/drive/MyDrive/Internship /LJSpeech-1.1/wavs"`)을 사용해 일반 음성과 비교 가능하도록 구성했습니다.

2. **Wav2Vec 2.0 제거**  
   초기에는 Wav2Vec2.0을 transfer learning 형태로 feature extractor로 사용하려 했지만, 모델이 너무 커서 속도와 실용성 면에서 부적합하다고 판단.  
   **정확도 차이가 크지 않다면, 실용적이고 빠른 모델이 상용화에 더 적합**하다고 생각하여 LCNN과 ResNetV2를 중심으로 비교 실험을 진행했습니다.

3. **데이터 전처리**
   많은 데이터 전처리 기법이 논문에 있었다. MFCC, Log-Mel, SpecAverage. 여기서 가장 정확한 전처리 방법을 찾아야 했었다.
   왜냐면 이미 Wav2Vec2.0를 제거 했기에 가능한 많은 feature extraction을 했어야 했다.  
   그리고 
   - MFCC
   - Log-Mel
   - SpecAverage
  
4. 딥러닝 사용(light version)
   깊이 있는 신경망이 있을 수록 학습을 더 잘할 수 있기에 이를 잘 할 수 있기에 LCNN 사용함.
   
---

## Part 3: Documentation & Analysis

1. Document your implementation process, including:
    - Any challenges encountered:
   1. 다운로드한 fake 오디오 데이터셋이 너무 커서 전처리와 로딩에 오랜 시간이 걸렸고, 모두 generated audio라서 일반화에 제한이 있을 수 있었습니다.
   2. Wav2Vec2.0은 powerful하긴 하지만, 크기가 커서 real-time 적용에는 부적합했습니다. 특히 모바일/경량화가 중요한 환경에서는 사용이 어려움.
  
    - How you addressed these challenges:
    - 1. 실제 데이터 셋을 구했다 - ("/content/drive/MyDrive/Internship /LJSpeech-1.1/wavs")코드에 있음),
    - 2. 초기에는 Wav2Vec2.0을 transfer learning 형태로 feature extractor로 사용하려 했지만, 모델이 너무 커서 속도와 실용성 면에서 부적합하다고 판단.
정확도 차이가 크지 않다면, 실용적이고 빠른 모델이 상용화에 더 적합하다고 생각하여 LCNN과 ResNetV2를 중심으로 비교 실험을 진행했습니다
   - LCNN 입력할 때, feature의 크기가 다름.
   - 새로 입력한 데이터셋 오디오 길이가 길었음.  
    - Assumptions made:  
  
3. Include an analysis section that addresses:
    - Why you selected this particular model for implementation: 결과를 보니 앙상블이 좋다는 효과를 봤다
    - How the model works (high-level technical explanation): SpecAverage 전처리하고 logMEL을 함. 원래는 MFCC를 하려고 했는데, 근데 너 같으면 왜 SpecAverge+logMel을 선택해? over MFCC? 
    - Performance results on your chosen dataset: 아직 모델 돌리는 중 
    - Observed strengths and weaknesses: 
    - Suggestions for future improvements: 

4. Reflection questions to address:
    1. What were the most significant challenges in implementing this model? 
    2. How might this approach perform in real-world conditions vs. research datasets?
    3. What additional data or resources would improve performance?
    4. How would you approach deploying this model in a production environment?
  
   확장자 ㅅㅂ....wav. WAV 차이가 있을지...

   🎯 문제 다시 정리:
real 오디오는 약 10초, fake 오디오는 약 5초
→ 모델이 "길이 차이"만으로 진짜/가짜를 분류할 위험
→ 일반화 성능 저하 가능

def extract_logmel(path, sr=16000, n_mels=N_MELS):
    y, _ = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    ...
    따라서 log-Mel도 시간 축(t) 길이가 서로 달라집니다.
    패딩: 짧은 오디오 파일 뒤에 0 값을 채워 길이를 맞추는 방법
    문제 있음: librosa.load()는 오디오 길이에 따라 그대로 로딩합니다.

즉, real은 10초 → 160,000 samples, fake는 5초 → 80,000 samples

따라서 log-Mel도 시간 축(t) 길이가 서로 달라집니다.

def pad_collate_fn(batch):
    features, labels = zip(*batch)
    max_len = max([f.shape[-1] for f in features])
    padded = [torch.nn.functional.pad(f, (0, max_len - f.shape[-1])) for f in features]
    return torch.stack(padded), torch.tensor(labels).float()
부분적으로 해결됨: 배치 단위에서는 spectrogram의 길이를 패딩으로 맞추고 있음

하지만 원래 길이에 따른 정보는 여전히 feature에 남아 있음

즉, 모델이 여전히 **"짧은 스펙트로그램이면 fake, 길면 real"**이라는 힌트를 학습할 가능성 있음
→ 학습이 길이에 bias될 수 있고, 이는 generalization에 취약점이 됩니다.


모든 오디오를 5초로 잘라서 고정 → 길이 정보가 모델에 반영되지 않음

진짜/가짜 모두 같은 길이로 비교되므로 길이에 의존한 분류 가능성 차단
✅ 결론
항목	현재 코드	해결 여부
오디오 길이 통일	❌ (librosa.load()로 원본 길이 유지)	해결 안 됨
길이 맞춤 처리	⭕ (pad_collate_fn으로 패딩)	일부 해결 (모양 맞춤만)
길이 bias 제거	❌ 모델이 여전히 길이 정보에 접근 가능	수정 필요


✅ 수정된 extract_logmel() 함수
python
복사
편집
def extract_logmel(path, sr=16000, n_mels=N_MELS, duration=5.0):
    y, _ = librosa.load(path, sr=sr)

    # 길이를 5초로 고정 (부족하면 0-padding, 넘치면 자르기)
    target_length = int(sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    # Mel → log-Mel 변환
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel)

    # SpecAverage augmentation
    log_mel_aug = spec_average_augment(log_mel)

    return torch.tensor(log_mel_aug).float()

    🔧 적용 방법
기존 코드에서 extract_logmel() 함수 전체를 위 내용으로 교체하세요.

호출하는 쪽(예: __getitem__)은 그대로 유지해도 작동합니다.

📌 이로 인해 얻는 효과
변경 전	변경 후
오디오 길이가 제각각 → 시간축 길이 다름 → 모델이 길이로 학습 가능성	오디오 길이를 5초로 고정 → 시간축도 일정 → 모델이 길이로 판단 불가
fake는 5초, real은 10초 → 무의식적 bias	동일한 길이에서 내용 기반 학습 가능



