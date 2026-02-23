# 전자상거래 고객 이탈 예측 시스템

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52.2-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

머신러닝 기반 고객 이탈 예측 및 리텐션 전략 제공 웹 애플리케이션

## 프로젝트 개요

고객 행동 데이터를 분석하여 이탈 위험을 사전 예측하고, 위험도별 맞춤형 마케팅 전략을 제공하는 시스템입니다.

**핵심 성과**: ROC-AUC 92.52% | Accuracy 91.60% | Precision 89.12% | Recall 80.80%

## 주요 기능

- **실시간 이탈 예측**: 15개 핵심 특성 기반 4단계 위험도 분류
- **모델 성능 비교**: 4개 ML 모델 평가 (XGBoost, Random Forest, Decision Tree, Logistic Regression)
- **맞춤형 전략 추천**: 위험도별 차별화된 마케팅 전략 자동 생성
- **인터랙티브 대시보드**: Streamlit 기반 웹 UI

## 기술 스택

```
Python 3.x
├── scikit-learn 1.7.2
├── xgboost 3.1.2
├── streamlit 1.52.2
├── pandas 2.3.3
├── numpy 2.2.6
├── plotly 6.5.0
├── matplotlib 3.10.0
├── seaborn 0.13.2
└── imbalanced-learn 0.14.1
```

## 빠른 시작

### 1. 설치

```bash
git clone https://github.com/jhwwon/ML-Project.git
cd ML-Project
pip install -r requirements.txt
```

### 2. 모델 학습

Jupyter Notebook에서 `ecommerce_churn_training_COMPLETE.ipynb` 실행

### 3. 웹앱 실행

```bash
streamlit run app_streamlit_COMPLETE.py
```

브라우저에서 `http://localhost:8501` 접속

## 데이터셋

- **출처**: [Kaggle E-Commerce Customer Behavior Dataset](https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset)
- **규모**: 10,000개 샘플, 24개 특성
- **타겟**: Churned (이탈 여부)

## 프로젝트 구조

```
Ecommerce_churn_MLProject/
├── dataset/
│   ├── ecommerce_customer_churn_dataset.csv
│   └── ecommerce_customer_churn_dataset_10k.csv
├── models/
│   └── churn_model_final.pkl
├── outputs/
├── ecommerce_churn_training_COMPLETE.ipynb
├── app_streamlit_COMPLETE.py
├── requirements.txt
└── README.md
```

## 핵심 구현

### 모델 성능 비교

| 모델 | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|------|---------|----------|-----------|--------|----------|
| **XGBoost** | **92.52%** | **91.60%** | **89.12%** | **80.80%** | **84.75%** |
| Random Forest | 92.40% | 91.05% | 85.95% | 82.53% | 84.20% |
| Decision Tree | 79.80% | 81.50% | 65.57% | 75.78% | 70.30% |
| Logistic Regression | 78.47% | 71.05% | 49.94% | 73.53% | 59.48% |

### Feature Selection
Random Forest 기반 특성 중요도 분석으로 24개 → 15개 특성 선택 (누적 중요도 87.53%)

### 클래스 불균형 해결
SMOTE 적용으로 Recall 32.7%p 향상 (클래스 비율 29:71 → 50:50)

### 하이퍼파라미터 튜닝
GridSearchCV + 5-Fold CV로 XGBoost 최적화

### 웹 애플리케이션
- 모델 캐싱 (`@st.cache_resource`)
- Session State 관리
- Plotly 인터랙티브 차트
- Glassmorphism 디자인

## 주요 특성 (Top 5)

1. **Customer_Service_Calls** (19.53%) - 고객 서비스 통화 수
2. **Cart_Abandonment_Rate** (12.16%) - 장바구니 이탈률
3. **Lifetime_Value** (11.68%) - 고객 평생 가치
4. **Total_Purchases** (8.22%) - 총 구매 횟수
5. **Discount_Usage_Rate** (7.82%) - 할인 사용률

## 라이선스

MIT License

## 연락처

**GitHub**: [@jhwwon](https://github.com/jhwwon)

---

**데이터셋 출처**: [Kaggle](https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset)
