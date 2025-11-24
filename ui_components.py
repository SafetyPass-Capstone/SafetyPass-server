# ui_components.py
"""오른쪽 판넬 함수"""

import streamlit as st

def render_system_status(G):
    """시스템 상태 패널 렌더링"""
    st.markdown("<h4 class='panel-title'>시스템 상태</h4>", unsafe_allow_html=True)
    
    total_seats = len([n for n in G.nodes if G.nodes[n]['type'] == 'seat'])
    total_capacity = sum(G.nodes[n].get('capacity', 0) for n in G.nodes if G.nodes[n]['type'] == 'seat')
    total_current = sum(G.nodes[n].get('current_people', 0) for n in G.nodes if G.nodes[n]['type'] == 'seat')
    total_exits = len([n for n in G.nodes if G.nodes[n]['type'] == 'exit'])
    
    # 메트릭 카드    
    # 층수, 출구 정보
    col_a, col_b, col_c= st.columns([1.5,1,1])
    with col_a:
        st.markdown(f"""
        <div class="metric-card" style="padding: 0.8rem;">
            <div class="metric-label" style="font-size: 0.75rem;">현재 관객 수</div>
            <div class="metric-value" style="font-size: 1.27rem;">{total_capacity:,}명</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown(f"""
        <div class="metric-card" style="padding: 0.8rem;">
            <div class="metric-label" style="font-size: 0.75rem;">층수</div>
            <div class="metric-value" style="font-size: 1.27rem;">{len(set(G.nodes[n].get('floor', 1) for n in G.nodes))}층</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_c:
        st.markdown(f"""
        <div class="metric-card" style="padding: 0.8rem;">
            <div class="metric-label" style="font-size: 0.75rem;">출구</div>
            <div class="metric-value" style="font-size: 1.27rem;">{total_exits}개</div>
        </div>
        """, unsafe_allow_html=True)

    # 대피 시간 정보 (화재 발생 후에만 표시)
    if (st.session_state.get('simulation_running', False) and 
        hasattr(st.session_state, 'total_evacuation_time') and 
        st.session_state.total_evacuation_time is not None):

        # 총 대피시간 (분:초 변환)
        total_min = int(st.session_state.total_evacuation_time)
        total_sec = int((st.session_state.total_evacuation_time * 60) % 60)

        # 평균 대피시간 (분:초 변환) - None 체크 추가
        if hasattr(st.session_state, 'avg_evacuation_time') and st.session_state.avg_evacuation_time is not None:
            avg_min = int(st.session_state.avg_evacuation_time)
            avg_sec = int((st.session_state.avg_evacuation_time * 60) % 60)
        else:
            avg_min = 0
            avg_sec = 0

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">총 대피시간</div>
            <div class="metric-value">{total_min}분 {total_sec}초</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">평균 대피시간</div>
            <div class="metric-value">{avg_min}분 {avg_sec}초</div>
        </div>
        """, unsafe_allow_html=True)


def render_exit_controls(G):
    """출구 제어 패널 (시각화 전용 - 경로 재계산 없음)"""
    from exit_control_handler import handle_exit_toggle_change, get_active_exits
    
    # 출구 노드 가져오기
    exit_nodes = sorted([n for n in G.nodes if G.nodes[n].get('type') == 'exit'], key=str)
    
    if not exit_nodes:
        st.warning("출구 노드가 없습니다.")
        return
    
    # 출구 상태 초기화
    if 'exit_status' not in st.session_state:
        st.session_state.exit_status = {str(n): True for n in exit_nodes}
    
    # 활성 출구 수 계산
    exit_status_dict = st.session_state.exit_status
    active_exits = get_active_exits(G, exit_status_dict)
    active_count = len(active_exits)
    
    # 박스 안에 제목과 출구 모두 포함
    with st.container():
        # 제목
        st.markdown("### 출구 제어")
        
        # 각 출구별 토글
        for exit_node in exit_nodes:
            exit_name = str(exit_node)
            
            # 2열 레이아웃: 출구명 | 토글
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{exit_name}**")
            
            with col2:
                # 현재 상태
                current_status = exit_status_dict.get(exit_name, True)
                
                # 고유 키 생성
                toggle_key = f"exit_toggle_{exit_name}"
                
                # 토글 버튼
                new_status = st.toggle(
                    f"출구 {exit_name}",
                    value=current_status,
                    key=toggle_key,
                    label_visibility="collapsed"
                )
                
                # ========== 수정: 경로 재계산 제거 ==========
                # 상태 변경 감지
                if new_status != current_status:
                    # 출구 상태만 업데이트 (경로 재계산 없음)
                    handle_exit_toggle_change(G, exit_name, new_status)
                    st.rerun()
                # ==========================================