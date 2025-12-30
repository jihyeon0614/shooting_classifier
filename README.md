# 🏀 Basketball Shot Classifier (농구 슛 폼 분류기)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Colab](https://img.shields.io/badge/Colab-Ready-orange)

## 📖 Project Overview (프로젝트 개요)
이 프로젝트는 **PyTorch**를 사용하여 **직접 설계한 CNN(Convolutional Neural Network) 모델**로 농구 슛 동작을 분류하는 AI입니다.
복잡한 사전 학습 모델을 사용하지 않고, 기초적인 **CNN 아키텍처를 밑바닥부터(Scratch) 구현**하여 이미지 분류의 기본 원리를 학습하고 적용했습니다.

### 🎯 주요 기능
입력된 이미지를 분석하여 다음 3가지 클래스 중 하나로 분류합니다:
1.  **Dunk Shoot** (덩크)
2.  **3-Point Shoot** (3점 슛)
3.  **Layup Shoot** (레이업)

---

## 🛠 Tech Stack (사용 기술)
* **Language:** Python
* **Framework:** PyTorch, Torchvision
* **Model Architecture:** Custom CNN (2 Convolutional Layers + 3 Fully Connected Layers)
* **Environment:** Google Colab

---

## 📊 Methodology (학습 방법)

### 1. Data Preprocessing (데이터 전처리)
* **Resize:** 모든 이미지를 (128, 128) 크기로 통일
* **Normalization:** 이미지를 Tensor로 변환 (0~1 사이 값)
* **DataLoader:** Batch size를 4로 설정하여 학습 효율화

### 2. Model Architecture (모델 구조)
가볍고 빠른 학습을 위해 **2개의 합성곱 층(Conv Layer)**과 **풀링 층(Pooling Layer)**을 교차하여 특징을 추출하고, 마지막에 분류를 수행하는 구조입니다.

* **Feature Extraction:**
    * `Conv2d` (3 -> 32 filters) + `ReLU` + `MaxPool`
    * `Conv2d` (32 -> 64 filters) + `ReLU` + `MaxPool`
* **Classification:**
    * `Flatten` (1차원으로 펼치기)
    * `Linear` (Fully Connected Layer)를 거쳐 최종 3개 클래스 확률 출력

```python
# 사용한 모델 구조 (ShootClassifier)
class ShootClassifier(nn.Module):
    def __init__(self):
        super(ShootClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 3) # 3 classes
basketball-classifier/
├── basketball_data/       # 학습 데이터 폴더
│   ├── train/
│   │   ├── dunk/
│   │   ├── layup/
│   │   └── three_point/
├── 슈팅예측.ipynb          # 메인 소스 코드 (Jupyter Notebook)
└── README.md              # 프로젝트 설명
🚀 How to Run (실행 방법)
이 프로젝트는 Google Colab 환경에 최적화되어 있습니다.

이 저장소를 Clone 하거나 다운로드합니다.

슈팅예측.ipynb 파일을 Google Colab에서 엽니다.

데이터셋을 준비하여 업로드합니다. (데이터 경로 수정 필요)

코드를 순차적으로 실행하여 모델 학습 및 테스트를 진행합니다.
train data set 다운로드 링크 : https://drive.google.com/drive/folders/1VEcPWDDyta-voxnDXX9Hn86meZwwFNOA?usp=drive_link
