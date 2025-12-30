"""
전자상거래 고객 이탈 예측 - Streamlit 웹 애플리케이션 (완전 개선 버전)
Feature Selection + SMOTE + 하이퍼파라미터 튜닝 적용
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os
import base64
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageFilter, ImageEnhance

# 페이지 설정
st.set_page_config(
    page_title="고객 이탈 예측 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Matplotlib 한글 폰트 설정
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    st.warning(f"Matplotlib 한글 폰트 설정 중 오류 발생: {e}")


# CSS 스타일 (전체 레이아웃 넓게, 이미지 등 스타일 포함)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&family=Noto+Sans+KR:wght@300;400;700&display=swap');

    * {
        font-family: 'Malgun Gothic', 'Outfit', 'Noto Sans KR', sans-serif;
    }

    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        font-weight: 300;
        margin-bottom: 3rem;
    }

    /* 히어로 배너 스타일 */
    .hero-section {
        position: relative;
        overflow: hidden;
        border-radius: 24px;
        margin-bottom: 3rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.08);
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
    }

    .glass-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 10px;
        text-shadow: 0 4px 10px rgba(0,0,0,0.2);
        letter-spacing: -1px;
    }

    .glass-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.95;
    }

    .block-container {
        max-width: 1300px !important;
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
    }

    /* 깔끔한 카드 스타일 */
    div.stForm {
        border-radius: 20px !important;
        border: 1px solid #f0f2f6 !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.03) !important;
        background-color: white !important;
        padding: 40px !important;
    }

    /* 이미지 패딩 제거를 위한 전역 설정 */
    [data-testid="stImage"] {
        margin-bottom: -7px !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """저장된 모델 로드"""
    # 가능한 모델 파일 경로들
    possible_paths = [
        'churn_model_final.pkl',
        'models/churn_model_final.pkl',
        'churn_model.pkl',
        'models/churn_model.pkl'
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        st.error(f"⚠️ 모델 파일을 찾을 수 없습니다!")
        st.info("📝 다음 경로 중 하나에 모델 파일을 배치해주세요:")
        for path in possible_paths:
            st.code(path)
        st.info("먼저 Jupyter 노트북(ecommerce_churn_training_COMPLETE.ipynb)을 실행하여 모델을 학습시켜주세요.")
        st.stop()
    
    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        st.success(f"✅ 모델 로드 성공: {model_path}")
        return model_package
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {str(e)}")
        st.stop()


def preprocess_input(data, model_package):
    """입력 데이터 전처리"""
    # DataFrame으로 변환
    df = pd.DataFrame([data])
    feature_names = model_package['feature_names']
    # 누락된 컬럼은 0으로 채움
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    # 순서 맞추기
    df = df[feature_names]
    # 스케일링
    scaler = model_package['scaler']
    scaled_data = scaler.transform(df)
    return scaled_data


def get_risk_level(churn_prob):
    """이탈 확률에 따른 위험도 분류"""
    if churn_prob >= 0.7:
        return {'level': '매우 높음', 'color': '#e74c3c', 'emoji': '🔴'}
    elif churn_prob >= 0.5:
        return {'level': '높음', 'color': '#f39c12', 'emoji': '🟠'}
    elif churn_prob >= 0.3:
        return {'level': '보통', 'color': '#3498db', 'emoji': '🟡'}
    else:
        return {'level': '낮음', 'color': '#2ecc71', 'emoji': '🟢'}


def get_recommendations(churn_prob, input_data):
    """이탈 확률에 따른 맞춤 추천"""
    recommendations = []
    
    if churn_prob >= 0.7:
        recommendations.append("🚨 **즉시 조치 필요**: VIP 할인 쿠폰 제공")
        recommendations.append("📞 **개인 상담**: 고객 서비스 팀 즉시 연락")
    elif churn_prob >= 0.5:
        recommendations.append("🎁 **특별 프로모션**: 맞춤형 할인 제안")
        recommendations.append("📧 **재참여 캠페인**: 이메일 마케팅 강화")
    elif churn_prob >= 0.3:
        recommendations.append("👀 **모니터링**: 정기적인 활동 추적")
        recommendations.append("💝 **로열티 프로그램**: 포인트 적립 혜택")
    else:
        recommendations.append("✅ **유지 관리**: 현재 만족도 유지")
        recommendations.append("🌟 **추천 요청**: 신규 고객 추천 유도")
    
    # 특정 지표 기반 추가 추천
    if input_data['Customer_Service_Calls'] > 5:
        recommendations.append("🆘 **서비스 개선**: 고객 불만 사항 해결")
    
    if input_data['Cart_Abandonment_Rate'] > 60:
        recommendations.append("🛒 **결제 프로세스 개선**: 장바구니 이탈 방지")
    
    if input_data['Days_Since_Last_Purchase'] > 60:
        recommendations.append("🔔 **재구매 유도**: 신제품 안내 및 할인")
    
    return recommendations


def main():
    # 모델 로드 (최상단에서)
    model_package = load_model()
    model_name = model_package.get('model_name', 'XGBoost')
    model_acc = model_package.get('accuracy', None)

    # 사이드바 구성 (모델 정보 포함)
    with st.sidebar:
        st.title("📊 고객 이탈 예측 시스템")
        st.markdown("고객 정보를 입력해 이탈 확률을 예측하고, 다양한 분석과 추천을 확인하세요.")
        st.markdown("---")
        st.subheader("모델 정보")
        st.markdown(f"- **모델명:** {model_name}")
        if model_acc is not None:
            st.markdown(f"- **정확도:** {model_acc:.2%}")
        st.markdown("- **Feature Selection** 및 **SMOTE** 적용")

    # 현대적인 슬림 히어로 배너 구현
    new_banner_path = "C:/Users/user/.gemini/antigravity/brain/7ef5c0fd-633b-4d2a-81aa-2b57880a0aae/modern_churn_analysis_banner_1767000434988.png"
    
    if os.path.exists(new_banner_path):
        st.markdown(f"""
            <div style="
                position: relative; 
                height: 220px; 
                border-radius: 24px; 
                overflow: hidden; 
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            ">
                <img src="data:image/png;base64,{base64.b64encode(open(new_banner_path, "rb").read()).decode()}" 
                     style="width: 100%; height: 100%; object-fit: cover; opacity: 0.9;">
                <div style="
                    position: absolute; 
                    top: 0; left: 0; width: 100%; height: 100%;
                    display: flex; justify-content: center; align-items: center;
                    background: linear-gradient(90deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
                ">
                    <div style="
                        background: rgba(255, 255, 255, 0.2);
                        backdrop-filter: blur(12px);
                        -webkit-backdrop-filter: blur(12px);
                        border: 1px solid rgba(255, 255, 255, 0.3);
                        padding: 25px 50px;
                        border-radius: 20px;
                        text-align: center;
                    ">
                        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #0f3d7a; letter-spacing: -1px; font-family: 'Malgun Gothic', sans-serif;">고객 이탈 예측 시스템</h1>
                        <p style="margin: 5px 0 0 0; font-size: 1rem; color: #444; font-weight: 400; opacity: 0.8;">Smart E-Commerce Analytics</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.title("📊 고객 이탈 예측 시스템")
        st.markdown("---")

    # 탭 UI
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["예측하기", "모델 분석", "맞춤 추천", "사용 가이드", "컬럼설명"])

    # 탭 1: 예측
    with tab1:
        st.header("고객 정보 입력")
        # Feature Selection 안내
        n_features = len(model_package['feature_names'])
        st.markdown(f"""
        <div class='info-box'>
            <strong>💡 Feature Selection 적용</strong><br>
            전체 특성 중 중요도 상위 <strong>{n_features}개</strong> 특성만 사용하여 예측 성능을 유지하면서 
            학습 속도를 크게 향상시켰습니다.
        </div>
        """, unsafe_allow_html=True)
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("🔴 위험 신호")
                customer_service_calls = st.number_input(
                    "고객 서비스 통화 수", min_value=0, value=3, help="높을수록 이탈 위험 증가 (1순위 중요도)"
                )
                cart_abandonment = st.number_input(
                    "장바구니 이탈률 (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1, help="높을수록 이탈 위험 증가 (3순위 중요도)"
                )
                days_since_last = st.number_input(
                    "마지막 구매 후 경과일", min_value=0, value=30, help="길수록 이탈 위험 증가 (7순위 중요도)"
                )
                returns_rate = st.number_input(
                    "반품률 (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1, help="높을수록 이탈 위험 증가 (13순위 중요도)"
                )
                discount_usage = st.number_input(
                    "할인 사용률 (%)", min_value=0.0, max_value=100.0, value=40.0, step=0.1, help="6순위 중요도"
                )
            with col2:
                st.subheader("💰 가치 지표")
                lifetime_value = st.number_input(
                    "평생 가치 ($)", min_value=0.0, value=2000.0, step=0.01, help="높을수록 중요한 고객 (2순위 중요도)"
                )
                total_purchases = st.number_input(
                    "총 구매 횟수", min_value=0.0, value=15.0, step=0.1, help="많을수록 충성 고객 (5순위 중요도)"
                )
                avg_order_value = st.number_input(
                    "평균 주문 금액 ($)", min_value=0.0, value=120.0, step=0.01, help="8순위 중요도"
                )
                credit_balance = st.number_input(
                    "크레딧 잔액 ($)", min_value=0.0, value=500.0, step=0.01, help="15순위 중요도"
                )
            with col3:
                st.subheader("📊 활동 지표")
                age = st.number_input(
                    "나이", min_value=18, max_value=100, value=35, help="4순위 중요도"
                )
                session_duration = st.number_input(
                    "평균 세션 시간 (분)", min_value=0.0, value=30.0, step=0.1, help="10순위 중요도"
                )
                pages_per_session = st.number_input(
                    "세션당 페이지 수", min_value=0.0, value=8.0, step=0.1, help="11순위 중요도"
                )
                email_open_rate = st.number_input(
                    "이메일 오픈률 (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1, help="9순위 중요도"
                )
                mobile_usage = st.number_input(
                    "모바일 앱 사용률 (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1, help="12순위 중요도"
                )
                login_frequency = st.number_input(
                    "로그인 빈도 (월간)", min_value=0, value=15, help="14순위 중요도"
                )
            submit_button = st.form_submit_button("🔮 이탈 확률 예측", use_container_width=True)
        if submit_button:
            # 입력 데이터 구성
            input_data = {
                'Customer_Service_Calls': customer_service_calls,
                'Lifetime_Value': lifetime_value,
                'Cart_Abandonment_Rate': cart_abandonment,
                'Age': age,
                'Total_Purchases': total_purchases,
                'Discount_Usage_Rate': discount_usage,
                'Days_Since_Last_Purchase': days_since_last,
                'Average_Order_Value': avg_order_value,
                'Email_Open_Rate': email_open_rate,
                'Session_Duration_Avg': session_duration,
                'Pages_Per_Session': pages_per_session,
                'Mobile_App_Usage': mobile_usage,
                'Returns_Rate': returns_rate,
                'Login_Frequency': login_frequency,
                'Credit_Balance': credit_balance
            }
            # 전처리
            processed_data = preprocess_input(input_data, model_package)
            # 예측
            model = model_package['model']
            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0]
            churn_prob = prediction_proba[1]
            retain_prob = prediction_proba[0]
            risk = get_risk_level(churn_prob)
            # 결과 표시
            st.markdown("---")
            st.header("🎯 예측 결과")
            col1, col2, col3 = st.columns(3)
            with col1:
                if prediction == 1:
                    st.markdown(
                        '<div class="prediction-box churn-box">'
                        '<h2>⚠️ 이탈 예상</h2>'
                        '<p style="font-size: 1.5rem;">고객이 떠날 가능성이 높습니다</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="prediction-box retain-box">'
                        '<h2>✅ 유지 예상</h2>'
                        '<p style="font-size: 1.5rem;">고객이 유지될 가능성이 높습니다</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
            with col2:
                st.metric("이탈 확률", f"{churn_prob*100:.2f}%", 
                         delta=f"{(churn_prob-0.5)*100:.1f}%p" if churn_prob > 0.5 else None,
                         delta_color="inverse")
                st.metric("유지 확률", f"{retain_prob*100:.2f}%",
                         delta=f"{(retain_prob-0.5)*100:.1f}%p" if retain_prob > 0.5 else None)
            with col3:
                st.metric("위험도", f"{risk['emoji']} {risk['level']}")
                st.markdown(f"<p style='color: {risk['color']}; font-size: 1.2rem; font-weight: bold;'>리스크 레벨</p>", 
                           unsafe_allow_html=True)
            # 확률 차트
            import uuid
            fig = go.Figure(data=[
                go.Bar(name='확률', 
                      x=['유지', '이탈'], 
                      y=[retain_prob*100, churn_prob*100],
                      marker_color=['#2ecc71', '#e74c3c'],
                      text=[f'{retain_prob*100:.1f}%', f'{churn_prob*100:.1f}%'],
                      textposition='auto')
            ])
            fig.update_layout(
                title="예측 확률 비교",
                yaxis_title="확률 (%)",
                showlegend=False,
                height=400,
                font=dict(family="Malgun Gothic")
            )
            chart_key = f"예측결과_{str(uuid.uuid4())}"
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            # 맞춤 추천 미리보기
            st.markdown("---")
            st.subheader("💡 빠른 추천")
            recommendations = get_recommendations(churn_prob, input_data)
            for i, rec in enumerate(recommendations[:3], 1):
                st.info(f"{i}. {rec}")
            if len(recommendations) > 3:
                st.caption("더 많은 추천은 '💡 맞춤 추천' 탭에서 확인하세요.")
            # 결과를 세션에 저장
            st.session_state['last_prediction'] = {
                'input_data': input_data,
                'churn_prob': churn_prob,
                'retain_prob': retain_prob,
                'risk': risk,
                'timestamp': datetime.now()
            }
            st.caption(f"⏰ 예측 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 탭 2: 모델 분석
    with tab2:
        st.header("📊 모델 성능 분석")
        results_df = pd.DataFrame(model_package['all_results'])
        # 컬럼 이름 정규화
        column_mapping = {
            'roc_auc': 'ROC-AUC',
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1-Score'
        }
        results_df = results_df.rename(columns=column_mapping)
        results_df = results_df.sort_values('ROC-AUC', ascending=False)
        # 성능 비교 차트
        metrics_to_plot = [col for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'] 
                          if col in results_df.columns]
        fig = px.bar(results_df, 
                    x='Model', 
                    y=metrics_to_plot,
                    title="모델별 성능 지표 비교",
                    barmode='group',
                    height=500)
        fig.update_layout(
            legend_title_text='성능 지표',
            font=dict(family="Malgun Gothic")
        )
        st.plotly_chart(fig, use_container_width=True, key="모델분석")
        # Feature Importance 시각화
        if 'feature_importance' in model_package:
            st.subheader("🎯 Feature Importance")
            fi = model_package['feature_importance']
            fig_fi = px.bar(
                x=fi['importances'], y=fi['features'],
                orientation='h', labels={'x': '중요도', 'y': '특성'},
                title="Feature Importance"
            )
            fig_fi.update_layout(
                yaxis={'categoryorder':'total ascending'},
                font=dict(family="Malgun Gothic")
            )
            st.plotly_chart(fig_fi, use_container_width=True, key="feature_importance")

        # Confusion Matrix 시각화
        if 'confusion_matrix' in model_package:
            st.subheader("🟦 Confusion Matrix")
            import seaborn as sns
            import matplotlib.pyplot as plt
            import platform

            # Matplotlib 한글 폰트 설정
            if platform.system() == 'Windows':
                plt.rc('font', family='Malgun Gothic')
            plt.rcParams['axes.unicode_minus'] = False
            
            cm = model_package['confusion_matrix']
            labels = model_package.get('confusion_labels', ['유지(0)', '이탈(1)'])
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                        xticklabels=labels, yticklabels=labels)
            ax.set_xlabel('예측값')
            ax.set_ylabel('실제값')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig_cm)

        # ROC Curve 시각화
        if 'roc_curve' in model_package:
            st.subheader("📈 ROC Curve")
            roc = model_package['roc_curve']
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=roc['fpr'], y=roc['tpr'], mode='lines', name='ROC Curve', line=dict(color='royalblue')))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
            fig_roc.update_layout(
                xaxis_title='False Positive Rate', 
                yaxis_title='True Positive Rate', 
                title='ROC Curve', 
                width=500, height=400,
                font=dict(family="Malgun Gothic")
            )
            st.plotly_chart(fig_roc, use_container_width=True, key="roc_curve")

        # Precision-Recall Curve 시각화
        if 'pr_curve' in model_package:
            st.subheader("📉 Precision-Recall Curve")
            pr = model_package['pr_curve']
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=pr['recall'], y=pr['precision'], mode='lines', name='PR Curve', line=dict(color='seagreen')))
            fig_pr.update_layout(
                xaxis_title='Recall', 
                yaxis_title='Precision', 
                title='Precision-Recall Curve', 
                width=500, height=400,
                font=dict(family="Malgun Gothic")
            )
            st.plotly_chart(fig_pr, use_container_width=True, key="pr_curve")
        # 상세 표
        st.subheader("📋 전체 모델 성능 비교")
        st.dataframe(results_df, hide_index=True, use_container_width=True)
        # 최고 성능 모델 하이라이트
        best_model = results_df.iloc[0]
        st.markdown("---")
        st.subheader("🏆 최고 성능 모델")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("모델", best_model['Model'])
        with col2:
            st.metric("ROC-AUC", f"{best_model['ROC-AUC']:.4f}")
        with col3:
            st.metric("Accuracy", f"{best_model['Accuracy']:.4f}")
        with col4:
            st.metric("F1-Score", f"{best_model['F1-Score']:.4f}")
    
    # 탭 3: 맞춤 추천
    with tab3:
        st.header("💡 맞춤형 이탈 방지 전략")
        
        if 'last_prediction' in st.session_state:
            pred = st.session_state['last_prediction']
            
            st.markdown(f"""
            <div class='info-box'>
                <strong>📊 마지막 예측 정보</strong><br>
                이탈 확률: <strong>{pred['churn_prob']*100:.2f}%</strong> | 
                위험도: <strong>{pred['risk']['emoji']} {pred['risk']['level']}</strong> | 
                예측 시각: {pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
            
            recommendations = get_recommendations(pred['churn_prob'], pred['input_data'])
            
            st.subheader("🎯 추천 전략")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")
            
            # 세부 분석
            st.markdown("---")
            st.subheader("🔍 세부 위험 요인 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**⚠️ 주요 위험 지표**")
                if pred['input_data']['Customer_Service_Calls'] > 5:
                    st.error(f"고객 서비스 통화: {pred['input_data']['Customer_Service_Calls']}회 (기준: 5회 이하)")
                if pred['input_data']['Cart_Abandonment_Rate'] > 60:
                    st.error(f"장바구니 이탈률: {pred['input_data']['Cart_Abandonment_Rate']:.1f}% (기준: 60% 이하)")
                if pred['input_data']['Days_Since_Last_Purchase'] > 60:
                    st.error(f"마지막 구매 후 경과: {pred['input_data']['Days_Since_Last_Purchase']}일 (기준: 60일 이하)")
            
            with col2:
                st.markdown("**✅ 긍정적 지표**")
                if pred['input_data']['Lifetime_Value'] > 1500:
                    st.success(f"평생 가치: ${pred['input_data']['Lifetime_Value']:.2f} (우수)")
                if pred['input_data']['Total_Purchases'] > 10:
                    st.success(f"총 구매 횟수: {pred['input_data']['Total_Purchases']:.1f}회 (우수)")
                if pred['input_data']['Email_Open_Rate'] > 30:
                    st.success(f"이메일 오픈률: {pred['input_data']['Email_Open_Rate']:.1f}% (우수)")
        
        else:
            st.info("먼저 '🔮 예측하기' 탭에서 예측을 실행해주세요.")
    
    # 탭 4: 사용 가이드
    with tab4:
        st.header("ℹ️ 사용 가이드")
        
        st.markdown("""
        ### 🚀 시작하기
        
        1. **모델 학습** (최초 1회만 필요)
           - `ecommerce_churn_training_COMPLETE.ipynb` 노트북 실행
           - 모든 셀을 순서대로 실행
           - `churn_model_final.pkl` 파일 생성 확인
        
        2. **웹 앱 실행**
           ```bash
           streamlit run app_streamlit.py
           ```
        
        3. **예측하기**
           - "🔮 예측하기" 탭에서 고객 정보 입력
           - "이탈 확률 예측" 버튼 클릭
           - 결과 및 추천 전략 확인
        
        ---
        
        ### 📞 문의사항
        
        프로젝트 관련 문의사항이나 개선 제안이 있으시면 언제든지 연락주세요!
        """)
        
        st.markdown("""
        ### ⚡ 개선사항
        
        이 버전은 다음과 같이 완전히 개선되었습니다:
        
        #### 📊 데이터 전처리
        - ✅ 결측치 처리 (중앙값/최빈값 대체)
        - ✅ 범주형 변수 인코딩 (Label Encoding)
        - ✅ 이상치 탐지 및 분석
        - ✅ 데이터 스케일링 (StandardScaler)
        
        #### 🎯 Feature Selection
        - ✅ Random Forest 기반 특성 중요도 분석
        - ✅ 상위 15개 핵심 특성 선택
        - ✅ 누적 중요도 ~86% 달성
        - ✅ 특성 수 37.5% 감소
        
        #### ⚖️ 클래스 불균형 처리
        - ✅ SMOTE 기법 적용
        - ✅ 학습 데이터 균형 맞춤
        
        #### 🔧 하이퍼파라미터 튜닝
        - ✅ GridSearchCV 적용
        - ✅ 최적 파라미터 탐색
        - ✅ 5-Fold 교차 검증
        
        #### 📈 모델 평가
        - ✅ 5개 모델 비교 평가
        - ✅ ROC Curve & PR Curve
        - ✅ 혼동 행렬 분석
        - ✅ Feature Importance 재분석
        
        ---
        
        ### 📊 15개 핵심 특성
        
        | 순위 | 특성명 | 중요도 | 설명 |
        |------|--------|--------|------|
        | 🥇 1 | Customer_Service_Calls | 12.60% | 고객 서비스 통화 수 |
        | 🥈 2 | Lifetime_Value | 12.24% | 고객 평생 가치 |
        | 🥉 3 | Cart_Abandonment_Rate | 9.40% | 장바구니 이탈률 |
        | 4 | Age | 6.17% | 나이 |
        | 5 | Total_Purchases | 5.64% | 총 구매 횟수 |
        | 6 | Discount_Usage_Rate | 5.61% | 할인 사용률 |
        | 7 | Days_Since_Last_Purchase | 5.10% | 마지막 구매 후 경과일 |
        | 8 | Average_Order_Value | 5.02% | 평균 주문 금액 |
        | 9 | Email_Open_Rate | 4.57% | 이메일 오픈률 |
        | 10 | Session_Duration_Avg | 4.22% | 평균 세션 시간 |
        | 11 | Pages_Per_Session | 3.56% | 세션당 페이지 수 |
        | 12 | Mobile_App_Usage | 3.47% | 모바일 앱 사용률 |
        | 13 | Returns_Rate | 3.12% | 반품률 |
        | 14 | Login_Frequency | 2.80% | 로그인 빈도 |
        | 15 | Credit_Balance | 2.51% | 크레딧 잔액 |
        
        ---
        
        ### 🎯 이탈 위험도 기준
        
        - 🟢 **낮음** (0-30%): 안정적인 고객, 현재 관계 유지
        - 🟡 **보통** (30-50%): 주의 관찰 필요, 정기 모니터링
        - 🟠 **높음** (50-70%): 적극적 개입 필요, 맞춤 프로모션
        - 🔴 **매우 높음** (70-100%): 즉시 조치 필요, VIP 혜택 제공
        
        ---
        
        ### 💡 활용 사례
        
        1. **마케팅 타겟팅**
           - 이탈 위험 고객에게 맞춤형 할인 쿠폰 발송
           - 위험도별 차별화된 마케팅 캠페인 실행
        
        2. **고객 세분화**
           - 이탈 확률 기반 고객 그룹 분류
           - 각 그룹별 최적화된 유지 전략 수립
        
        3. **예방적 고객 관리**
           - 조기 경고 시스템으로 활용
           - 이탈 징후 발견 시 선제적 대응
        
        4. **리소스 최적화**
           - 고위험 고객에게 집중 투자
           - 효율적인 고객 유지 비용 관리
        
        ---
        
        ### � 문의사항
        
        프로젝트 관련 문의사항이나 개선 제안이 있으시면 언제든지 연락주세요!
        """)

    # 탭 5: 컬럼설명
    with tab5:
        st.header("📋 데이터셋 컬럼 상세 설명")
        st.markdown("분석에 사용된 전체 데이터셋의 컬럼 정보입니다.")
        
        column_data = [
            {"컬럼명": "Age", "설명": "고객의 연령 (세)"},
            {"컬럼명": "Gender", "설명": "고객의 성별 (Male / Female / Other)"},
            {"컬럼명": "Country", "설명": "고객 거주 국가"},
            {"컬럼명": "City", "설명": "고객 거주 도시"},
            {"컬럼명": "Membership_Years", "설명": "서비스 가입 기간 (연수)"},
            {"컬럼명": "Login_Frequency", "설명": "월 평균 로그인 빈도"},
            {"컬럼명": "Session_Duration_Avg", "설명": "평균 세션 유지 시간 (분)"},
            {"컬럼명": "Pages_Per_Session", "설명": "세션당 평균 페이지 조회 수"},
            {"컬럼명": "Cart_Abandonment_Rate", "설명": "장바구니 이탈률 (담기 후 미결제 비율, %)"},
            {"컬럼명": "Wishlist_Items", "설명": "관심 상품(위시리스트) 등록 개수"},
            {"컬럼명": "Total_Purchases", "설명": "누적 주문 횟수"},
            {"컬럼명": "Average_Order_Value", "설명": "주문당 평균 결제 금액 ($)"},
            {"컬럼명": "Days_Since_Last_Purchase", "설명": "마지막 구매 이후 경과일"},
            {"컬럼명": "Discount_Usage_Rate", "설명": "전체 구매 중 할인을 사용한 비율 (%)"},
            {"컬럼명": "Returns_Rate", "설명": "구매한 상품의 반품률 (%)"},
            {"컬럼명": "Email_Open_Rate", "설명": "마케팅 이메일을 확인한 비율 (%)"},
            {"컬럼명": "Customer_Service_Calls", "설명": "고객 센터 상담 및 문의 횟수"},
            {"컬럼명": "Product_Reviews_Written", "설명": "지금까지 작성한 상품 리뷰 총 개수"},
            {"컬럼명": "Social_Media_Engagement_Score", "설명": "브랜드 SNS 활동 지수"},
            {"컬럼명": "Mobile_App_Usage", "설명": "모바일 앱 사용 비중 및 적극성 점수"},
            {"컬럼명": "Payment_Method_Diversity", "설명": "사용한 결제 수단의 종류 수"},
            {"컬럼명": "Lifetime_Value", "설명": "고객 생애 가치 (현재까지 총 기여 수익, $)"},
            {"컬럼명": "Credit_Balance", "설명": "계정에 남은 크레딧/포인트 잔액 ($)"},
            {"컬럼명": "Signup_Quarter", "설명": "고객이 최초 가입한 분기 (Q1~Q4)"},
            {"컬럼명": "Churned", "설명": "이탈 여부 (1: 이탈, 0: 유지) - 예측 목표 변수"}
        ]
        
        # 데이터프레임으로 변환하여 표시
        df_cols = pd.DataFrame(column_data)
        st.table(df_cols)
        
        st.info("💡 위 컬럼들 중 중요도 분석을 통해 핵심적인 15개 특성이 모델 예측에 사용됩니다.")


if __name__ == "__main__":
    main()
