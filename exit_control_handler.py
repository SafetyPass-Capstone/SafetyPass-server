"""출구 제어 및 경로 재계산 핸들러"""
import streamlit as st
from utils import add_log


def get_active_exits(G, exit_status_dict):
    """
    활성화된 출구만 필터링 (시각화용)
    """
    all_exits = [n for n in G.nodes if G.nodes[n].get('type') == 'exit']
    
    active_exits = []
    for exit_node in all_exits:
        exit_name = str(exit_node)
        is_active = exit_status_dict.get(exit_name, True)
        if is_active:
            active_exits.append(exit_node)
    
    return active_exits


def handle_exit_toggle_change(G, exit_name, new_status):
    """
    출구 토글 변경 처리 - 시각화만 영향
    
    Args:
        G: NetworkX 그래프
        exit_name: 출구 이름
        new_status: 새로운 상태 (True/False)
    """
    # 상태 업데이트 (시각화용)
    if 'exit_status' not in st.session_state:
        st.session_state.exit_status = {}
    
    st.session_state.exit_status[exit_name] = new_status
    
    status_text = "활성화" if new_status else "비활성화"
    add_log(f"출구 {exit_name} {status_text} (시각화 전용)")



def validate_exit_status(G):
    """
    출구 상태 검증 (시각화용)
    
    Returns:
        bool: 유효한 상태인지 여부
    """
    exit_status_dict = st.session_state.get('exit_status', {})
    active_exits = get_active_exits(G, exit_status_dict)
    
    return len(active_exits) > 0