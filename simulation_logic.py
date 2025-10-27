# simulation_logic.py
"""시뮬레이션 로직"""

import streamlit as st
from collections import defaultdict
from utils import add_log

def run_simulation_step():
    """시뮬레이션 한 스텝 실행"""
    if not hasattr(st.session_state, 'people') or not st.session_state.people:
        return
    
    # node_capacity 초기화
    if not hasattr(st.session_state, 'node_capacity'):
        st.session_state.node_capacity = defaultdict(lambda: 3)
    
    # 각 노드에 이번 tick 동안 들어온 사람 수 초기화
    node_usage = defaultdict(int)
    
    for p in st.session_state.people:
        if p['done']:
            continue
        
        curr_node = p['path'][p['idx']]
        
        # 도착 여부 체크
        if p['idx'] == len(p['path']) - 1:
            p['done'] = True
            p['time'] = st.session_state.current_time
            continue
        
        next_node = p['path'][p['idx'] + 1]
        
        # 다음 노드에 여유가 있으면 이동
        if node_usage[next_node] < st.session_state.node_capacity[next_node]:
            node_usage[next_node] += 1
            p['idx'] += 1
    
    st.session_state.current_time += 1

def check_simulation_complete():
    """시뮬레이션 완료 여부 확인"""
    all_done = all(p['done'] for p in st.session_state.people) if hasattr(st.session_state, 'people') else False
    
    if all_done or st.session_state.current_time >= st.session_state.max_time:
        st.session_state.simulation_running = False
        
        # 대피 완료 시간 계산
        if hasattr(st.session_state, 'people') and st.session_state.people:
            all_times = [p['time'] for p in st.session_state.people]
            max_time = max(all_times)
            avg_time = sum(all_times) / len(all_times)
            
            st.session_state.evacuation_complete_time = max_time
            st.session_state.avg_evacuation_time = avg_time
        else:
            st.session_state.evacuation_complete_time = st.session_state.current_time
            st.session_state.avg_evacuation_time = st.session_state.current_time
        
        add_log(f"시뮬레이션 완료! 총 대피시간: {st.session_state.evacuation_complete_time / 60:.1f}분 (최대 대피시간)")
        
        return True
    
    return False