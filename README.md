# 🛡️ ChurnGuard AI : B2B 고객 이탈 예측 & 방어 대시보드

> **엑셀(CSV) 업로드 하나로 끝내는 실시간 고객 이탈 예측 & 방어 자동화 대시보드**  
> 초경량 딥러닝 신경망 엔진 기반 피처 엔지니어링 & 4대 핵심 경영 KPI 실시간 시각화 솔루션

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churnguard-web.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

---

## 📌 주요 핵심 기능 (Key Features)

1. **⚡ 초경량 고속 딥러닝 추론 엔진**
   * 독자 개발된 15KB 순수 NumPy 신경망 엔진 탑재
   * 대규모 클라우드 라이브러리 없이도 **소수점 7자리까지 100% 오차 없는 정밀 딥러닝 추론** 제공
   * **검증 적중률 82.38%** 의 신뢰도 높은 이탈 가능성 예측

2. **📊 경영진을 위한 4대 핵심 KPI 메트릭**
   * 실시간 모니터링 고객 수 및 고위험 이탈군 비율 산출
   * 이탈 방치 시 예상 연간 손실 매출 자동 계산
   * AI 골든타임 프로모션 집행 시 보존 기대 매출 정량화

3. **🎯 스마트 고객 필터링 & 액션 가이드**
   * 🔴 **고위험군 (70%+):** 긴급 VIP 전담 케어 & 즉각적인 방어 쿠폰 제안
   * 🟡 **중위험군 (40~70%):** 재방문 리마인드 알림톡 & 마일리지 적립 혜택
   * 🟢 **안전군 (40% 미만):** 불필요한 마케팅 비용 절감 대상
   * 원클릭 엑셀 호환 CSV 다운로드 (`UTF-8-SIG`)

4. **🔍 1:1 고객 맞춤 What-If 시뮬레이터**
   * 할인율, 최근 방문일, 구매액 등 주요 변수를 슬라이더로 조작하여 실시간 이탈률 변화를 즉각 시뮬레이션

5. **📱 모바일 100% 반응형 최적화**
   * 모바일 접속 시에도 2x2 반응형 그리드 레이아웃으로 쾌적한 분석 지원

---

## 🚀 로컬 실행 방법 (Quick Start)

```bash
# 1. 저장소 복제
git clone https://github.com/jchlee428-cyber/churnguard-web.git
cd churnguard-web

# 2. 가상환경 생성 및 의존성 패키지 설치
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# 3. Streamlit 대시보드 실행
.venv/Scripts/streamlit run app.py
```

브라우저에서 `http://localhost:8501`에 접속하여 사용하실 수 있습니다.

---

## 📂 배포 구조 (Structure)

```text
├── app.py                     # Streamlit 기반 ChurnGuard AI 메인 웹 대시보드
├── model_weights.npz          # 15KB 초경량 딥러닝 신경망 가중치
├── preprocessor_params.json   # 스케일러 및 전처리 파라미터
├── requirements.txt           # 클라우드 필수 의존성 패키지
├── LICENSE                    # 독점 소프트웨어 라이선스 (All Rights Reserved)
└── README.md                  # 대시보드 공식 소개 문서
```

---

## 📄 라이선스 & 문의 (License & Inquiries)

* 본 소프트웨어 및 AI 신경망 모델은 **독점 소프트웨어(Proprietary Software)** 로서 무단 상업적 복제, 재배포, 리버스 엔지니어링이 엄격히 금지됩니다.
* 비즈니스 도입, 제휴 및 맞춤형 컨설팅 문의: `jchlee428@gmail.com`
