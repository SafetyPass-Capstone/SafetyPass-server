# session_state.py
"""세션 상태 초기화 및 관리"""

import streamlit as st

def initialize_session_state():
    """세션 상태 초기화"""
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'current_time' not in st.session_state:
        st.session_state.current_time = 0
    if 'fire_node' not in st.session_state:
        st.session_state.fire_node = None
    if 'logs' not in st.session_state:
        st.session_state.logs = [
            f"시스템 초기화 완료",
        ]
    if 'gnn_model_loaded' not in st.session_state:
        st.session_state.gnn_model_loaded = False
    if 'evacuation_paths' not in st.session_state:
        st.session_state.evacuation_paths = None
    if 'bottleneck_data' not in st.session_state:
        st.session_state.bottleneck_data = None
    if 'venue_type' not in st.session_state:
        st.session_state.venue_type = 'I형'
    # venue_type 변경 감지 및 캐시 초기화
    if st.session_state.get('prev_venue_type') != st.session_state.venue_type:
        st.session_state.seat_units_df = None
        st.session_state.I_layer_geojson = None
        st.session_state.T_layer_geojson = None
        st.session_state.gnn_model_loaded = False
        st.session_state.evacuation_paths = {}
        st.session_state.bottleneck_data = {}
        if 'graph_data' in st.session_state:
            del st.session_state.graph_data
        if 'G' in st.session_state:
            del st.session_state.G
        st.session_state.prev_venue_type = st.session_state.venue_type
    if 'max_time' not in st.session_state:
        st.session_state.max_time = 600  # 10분
    if 'evacuation_complete_time' not in st.session_state:
        st.session_state.evacuation_complete_time = None
    if 'avg_evacuation_time' not in st.session_state:
        st.session_state.avg_evacuation_time = None