# 전자상거래 고객 이탈 예측 시스템

## 주요 기술
Python, Scikit-learn, XGBoost, Streamlit

## 프로젝트 개요
머신러닝 기반 고객 이탈 예측 시스템. Feature Selection + SMOTE + 하이퍼파라미터 튜닝을 적용하여 ROC-AUC 98.23% 달성. Streamlit 웹 대시보드를 통해 실시간 예측 및 맞춤 리텐션 전략 제공

## 주요 기능
- 이탈 확률 실시간 예측 - 15개 핵심 특성 기반 4단계 위험도 분류 (낮음/보통/높음/매우 높음)
- 5개 모델 성능 비교 - XGBoost, Random Forest, Logistic Regression, SVM, KNN 평가
- Feature Importance 시각화 - 고객 서비스 통화 수, 평생 가치, 장바구니 이탈률 등 주요 지표 분석
- 맞춤형 추천 전략 - 위험도별 차별화된 마케팅 전략 자동 생성
- Streamlit 웹 대시보드 - Glassmorphism 디자인, 인터랙티브 차트, 5개 탭 구조

## 핵심 구현
- Random Forest 기반 특성 중요도 분석으로 24개→15개 특성 선택 (누적 중요도 86% 유지)
- SMOTE로 클래스 불균형 해결 (29:71 → 50:50), Recall 32.7%p 향상
- GridSearchCV + 5-Fold CV로 XGBoost 하이퍼파라미터 최적화
- 5개 모델 비교 후 XGBoost 선정 (ROC-AUC 98.23%)
- Streamlit 웹앱 구현 - 모델 캐싱, 세션 상태 관리, Plotly 인터랙티브 차트

## Trouble Shooting

### 한글 폰트 깨짐
- Matplotlib 차트에서 한글이 □□□로 표시되는 문제
- try-except로 Malgun Gothic 설정, unicode_minus False 처리로 해결

### 모델 파일 경로 오류
- 실행 환경마다 모델 파일 경로가 달라 FileNotFoundError 발생
- 4가지 가능 경로('churn_model_final.pkl', 'models/churn_model_final.pkl' 등)를 순회하며 자동 탐색으로 해결

### Streamlit 차트 키 중복
- 페이지 rerun 시 동일 차트 키로 인한 충돌 오류
- uuid로 고유 키 생성(f"예측결과_{uuid.uuid4()}")으로 해결

### Feature 순서 불일치
- 학습 시와 예측 시 특성 순서가 달라 잘못된 예측 발생
- 모델 저장 시 feature_names 포함, 예측 시 df[feature_names] 순서 강제로 해결

## 결과 및 회고

**최종 성능**: ROC-AUC 98.23% | Accuracy 95.41% | Precision 95.89% | Recall 94.82%

머신러닝 파이프라인 전체 과정(전처리→학습→평가→배포)을 경험했으며, 단순 모델 정확도보다 **실무 적용성**이 중요함을 깨달았습니다. Feature Selection으로 특성 수를 줄이면서도 성능을 유지하는 경험, SMOTE로 불균형 데이터를 해결하는 과정, GridSearchCV로 체계적으로 하이퍼파라미터를 튜닝하는 방법을 익혔습니다. 특히 Jupyter Notebook에서 끝나지 않고 **Streamlit 웹앱으로 배포**하여 실사용 가능한 시스템을 만든 경험이 가장 값졌습니다.

**GitHub Code URL**  
https://github.com/yourusername/ML-Project
