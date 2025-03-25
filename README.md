# JAEYEON-JO-Momenta

## 🔍 Audio Deepfake Detection Models Comparison Table

| Model       | Key Technical Innovation | Reported Performance | Why It's Promising | Potential Limitations |
|-------------|---------------------------|-----------------------|----------------------|------------------------|
| **ResNet2** | - Residual learning structure<br>- Can incorporate dilated convolution & attention pooling | - EER ≤ **1.6%** (ADD 2022)<br>- Accuracy **98%+** | - Strong deep feature extraction<br>- Robust across varied conditions<br>- Well-suited for conversation-based detection | - High computational cost<br>- Limited ability to capture long-term context (needs Transformer integration) |
| **LCNN**    | - Lightweight CNN with Max-Feature-Map (MFM)<br>- Effective noise suppression | - EER around **3–5%**<br>- Real-time capable | - Excellent for **real-time detection**<br>- Suitable for edge devices<br>- Performs well in noisy environments | - Lacks deep contextual understanding when used alone<br>- Not ideal for complex conversations |
| **Wav2Vec2.0** | - Self-supervised learning<br>- Raw audio input + Transformer-based architecture | - EER ≤ **2.2%**<br>- Accuracy **97%+** | - Outstanding at capturing long context, emotion, and meaning<br>- Performs well with **limited labeled data**<br>- Ideal for real conversation analysis | - Large model size and high compute cost<br>- Requires optimization for real-time deployment |

---

### ✅ Summary Table 

| Model       | Accuracy | EER     | Real-time Capability | Conversational Analysis |
|-------------|----------|---------|------------------------|--------------------------|
| ResNet2     | 98%+     | ≤1.6%   | ⚠️ Moderate (needs optimization) | ✅ Highly suitable |
| LCNN        | ~95%     | 3–5%    | ✅ Very high           | ⚠️ Limited |
| Wav2Vec2.0  | 97%+     | ≤2.2%   | ❌ Needs high resources | ✅ Best performance |


## 2. 📊 Model Analysis

### 🤖 Models Considered

| Model        | Key Features                                      | EER      | Real-Time | Contextual Strength |
|--------------|---------------------------------------------------|----------|------------|----------------------|
| **LCNN**     | Lightweight CNN + Max-Feature-Map                 | 3–5%     | ✅ High     | ⚠️ Limited           |
| **ResNet2**  | Residual learning + dilated conv + attention pooling | ≤1.6% | ⚠️ Moderate | ✅ Strong            |
| **Wav2Vec2.0** | Transformer-based, raw waveform input            | ≤2.2%    | ❌ No       | ✅ Excellent         |

---

### 🔍 Why LCNN?

- ✅ Real-time capable  
- ✅ Easy to deploy on edge devices  
- ✅ Compatible with log-Mel features  
- ✅ Performs well with low-compute environments  

---

### 🔧 Preprocessing Strategy

| Method        | Pros                                      | Cons                               |
|---------------|-------------------------------------------|------------------------------------|
| **MFCC**      | Compact, traditional                      | May lose fine-grained patterns     |
| **Log-Mel**   | Preserves spectro-temporal details        | Needs augmentation to generalize   |
| **SpecAverage** | Adds regularization via masking          | Requires careful tuning            |

> **Final Choice**: `log-Mel + SpecAverage`, fixed to **5-second duration**  
> → Prevents length-based bias and encourages content-based learning

---

## 3. 📈 Performance (Ongoing)

Model training is currently in progress. Early observations suggest:

- ⚡ Fast convergence (LCNN overfits quickly → needs regularization)
- 🎯 Accuracy reaches 100% too early → dataset still small and simple

### 🔍 Metrics to be Evaluated:

- Accuracy  
- AUC (ROC)  
- EER (Equal Error Rate)

---

## 4. 🧠 Reflections

### Q1: What were the most significant challenges?

- Ensuring **model generalization**
- Removing **length-based bias**
- Balancing **practicality (speed)** vs. **performance (accuracy)**

---

### Q2: How might this approach perform in the real world?

- Requires diverse and noisy datasets to generalize well  
- LCNN is feasible for **real-time deployment**  
- Wav2Vec2.0 may need **distillation or pruning** to be practical

---

### Q3: What additional data would improve performance?

- Noisy real-world environments  
- Non-English speakers  
- Newer fake generation methods: **Vall-E, VITS, Bark**, etc.

---

### Q4: How would you deploy this?

- Export as **ONNX** or **TorchScript**  
- Apply **5-second sliding window inference**  
- Deploy in **real-time streaming pipelines** (e.g., call center audio monitoring)

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



