# utils.py
"""유틸리티 함수"""

import streamlit as st
from datetime import datetime
from config import CROWD_TYPE_NAMES, CROWD_TYPE_COLORS, BOTTLENECK_COLORS

# crowd_type 안전 처리 헬퍼 함수
def get_safe_crowd_type(node_data):
    """노드의 crowd_type을 안전하게 반환 (0, 1, 2만 허용)"""
    crowd_type = node_data.get('crowd_type', 1)
    if crowd_type not in [0, 1, 2]:
        return 1  # 기본값
    return crowd_type

def get_safe_crowd_color(node_data):
    """노드의 crowd_type에 해당하는 색상을 안전하게 반환"""
    crowd_type = get_safe_crowd_type(node_data)
    return CROWD_TYPE_COLORS[crowd_type]

def get_safe_crowd_name(node_data):
    """노드의 crowd_type에 해당하는 이름을 안전하게 반환"""
    crowd_type = get_safe_crowd_type(node_data)
    return CROWD_TYPE_NAMES[crowd_type]

def add_log(message):
    """로그 메시지 추가"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    st.session_state.logs.append(f"[{timestamp}] {message}")
    if len(st.session_state.logs) > 100:
        st.session_state.logs.pop(0)

def get_bottleneck_color(bottleneck_value):
    """병목도에 따른 색상 반환"""
    if bottleneck_value >= 0.20:
        return BOTTLENECK_COLORS['very_high']
    elif bottleneck_value >= 0.10:
        return BOTTLENECK_COLORS['high']
    elif bottleneck_value >= 0.05:
        return BOTTLENECK_COLORS['medium']
    elif bottleneck_value >= 0.02:
        return BOTTLENECK_COLORS['low']
    else:
        return BOTTLENECK_COLORS['very_low']