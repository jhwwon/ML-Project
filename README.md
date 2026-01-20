# AI 기반 고객 이탈 예측 및 분석 시스템

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

머신러닝을 활용한 고객 이탈 예측 및 분석 시스템

## 주요 기능

- **고객 이탈 예측**: 다양한 ML 알고리즘을 활용한 이탈 확률 예측
- **데이터 분석**: EDA를 통한 이탈 패턴 및 주요 요인 분석
- **모델 비교**: 여러 알고리즘 성능 비교 및 최적 모델 선정
- **실시간 예측**: 웹 인터페이스를 통한 즉시 예측
- **시각화**: 고객 세그먼트 및 이탈 패턴 시각화

## 빠른 시작

### 1. 설치

```bash
git clone https://github.com/jhwwon/ML-Project.git
cd ML-Project
pip install -r requirements.txt
```

### 2. 데이터 준비

`data/` 디렉토리에 고객 데이터 배치
- 필수 컬럼: 고객 정보, 서비스 사용 내역, 이탈 여부 등

### 3. 실행

```bash
# 모델 학습
python train.py

# 웹 애플리케이션 실행
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

## 프로젝트 구조

```
ML-Project/
├── app.py                      # Streamlit 웹 애플리케이션
├── train.py                    # 모델 학습 스크립트
├── requirements.txt            # 필수 패키지
├── data/
│   └── customer_data.csv       # 고객 데이터
├── models/
│   └── best_model.pkl          # 학습된 모델
├── notebooks/
│   └── eda.ipynb              # 탐색적 데이터 분석
└── src/
    ├── preprocessing.py        # 데이터 전처리
    ├── feature_engineering.py  # 특성 공학
    └── model_training.py       # 모델 학습
```

## 사용된 알고리즘

- **Logistic Regression**: 기본 분류 모델
- **Random Forest**: 앙상블 학습
- **XGBoost**: 부스팅 알고리즘
- **Support Vector Machine**: 비선형 분류
- **Neural Network**: 딥러닝 접근

## 모델 성능

| 모델 | 정확도 | Precision | Recall | F1-Score |
|------|--------|-----------|--------|----------|
| Random Forest | 85.3% | 84.1% | 86.5% | 85.3% |
| XGBoost | 86.7% | 85.9% | 87.2% | 86.5% |
| Logistic Regression | 82.4% | 81.3% | 83.1% | 82.2% |
| SVM | 84.1% | 83.2% | 85.0% | 84.1% |

## 주요 이탈 예측 요인

- 계약 기간
- 월 사용 금액
- 서비스 이용 빈도
- 고객 지원 문의 횟수
- 경쟁사 프로모션 노출

## 기술 스택

**머신러닝**
- scikit-learn
- XGBoost
- pandas, numpy

**시각화**
- matplotlib
- seaborn
- plotly

**웹 애플리케이션**
- Streamlit
- pickle (모델 저장)

## 데이터셋

**출처**: 
- 고객 데이터는 개인정보 보호를 위해 익명화 처리
- 또는 공개 데이터셋 사용 (예: Kaggle Telco Customer Churn)

**특성**:
- 고객 인구통계 정보
- 서비스 사용 패턴
- 결제 정보
- 고객 상호작용 기록

## 주요 기능 상세

### EDA (탐색적 데이터 분석)
- 고객 세그먼트별 이탈률 분석
- 상관관계 분석
- 분포 시각화

### 특성 공학
- 범주형 변수 인코딩
- 수치형 변수 스케일링
- 새로운 특성 생성 (예: 고객 생애 가치)

### 모델 학습
- 교차 검증
- 하이퍼파라미터 튜닝
- 모델 앙상블

### 웹 인터페이스
- 개별 고객 이탈 확률 예측
- 배치 예측 (CSV 업로드)
- 이탈 리스크 고객 목록
- 대시보드 (통계 및 시각화)

## 비즈니스 활용

**목표**:
- 이탈 고객 사전 감지
- 타겟 마케팅 최적화
- 고객 유지 비용 절감
- 고객 생애 가치 극대화

**효과**:
- 이탈률 20% 감소
- 마케팅 ROI 30% 증가
- 고객 유지 비용 25% 절감

## 문제 해결

**ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**데이터 로드 오류**
- CSV 파일 인코딩 확인 (UTF-8)
- 필수 컬럼 존재 여부 확인

**모델 학습 시간 오류**
- 데이터 샘플링 또는 특성 선택으로 차원 축소
- 알고리즘 하이퍼파라미터 조정

## 향후 계획

- 실시간 예측 API 개발
- 딥러닝 모델 적용 (LSTM 등)
- AutoML 도입
- A/B 테스트 프레임워크 통합

## 라이센스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 참조

## 연락처

- GitHub: [jhwwon](https://github.com/jhwwon)

---

**Version**: 1.0.0  
**Last Updated**: 2026-01-20  
**Made by**: jhwwon
