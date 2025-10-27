
# 다크 테마 CSS
DARK_THEME_CSS = """
<style>
    /* 전체 배경 - Streamlit 전역 컨테이너를 모두 어둡게 */
    html, body, .stApp, .stMain, .main,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    .block-container {
        background-color: #0e1117 !important;
    }

    /* 1) 기본: 거의 모든 텍스트를 흰색으로 고정 */
    .stApp,
    .stApp * {
        color: #ffffff !important;
    }


    /* 헤더 스타일 */
    .main-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ffffff !important;
        text-align: left;
        margin-bottom: 1rem;
    }

    .sub-header {
        font-size: 0.9rem;
        color: #8b949e;
        text-align: left;
        margin-bottom: 2rem;
    }

    /* 패널 제목: 시스템 상태, 범례 등 */
    .panel-title {
        color: #ffffff !important;
        margin: 0 0 0.75rem 0;
    }

    /* 3) 하지만 셀렉트박스(드롭다운) 내부 글씨는 제외 — 기본 색상으로 되돌림 */
    /* native select/option */
    select,
    select * ,
    option {
        color: initial !important;
        background-color: initial !important;
    }

    
    /* 메트릭 카드 */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16192b 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #8b949e;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    
    /* 상태 표시 */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .status-normal {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
    }
    
    .status-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
    }
    
    .status-danger {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    /* 로그 스타일 */
    .log-container {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .log-entry {
        color: #8b949e;
        margin: 0.3rem 0;
        padding: 0.2rem 0;
        border-bottom: 1px solid #161b22;
    }
    
    .log-time {
        color: #58a6ff;
        font-weight: bold;
    }
    
    /* 버튼 스타일 */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.8rem;
        font-weight: bold;
        font-size: 1rem;
        box-shadow: 0 4px 6px rgba(255, 68, 68, 0.3);
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        box-shadow: 0 6px 12px rgba(255, 68, 68, 0.5);
        transform: translateY(-2px);
    }
    
    /* 사이드바 */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
    /* 사이드바 텍스트 가독성 향상: 모든 텍스트를 흰색으로 고정 */
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] [class^="css-"] {
        color: #ffffff !important;
    }
    
    /* 라디오 버튼 */
    .stRadio > label {
        color: #c9d1d9;
        font-weight: 600;
    }
    
    /* 슬라이더 */
    .stSlider {
        padding: 1rem 0;
    }
</style>
"""