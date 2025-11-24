# sidebar.py
"""사이드바 UI 모듈"""

import streamlit as st
import random
from utils import add_log

def render_system_log():
    """시스템 로그 렌더링"""
    st.markdown("<h4 class='panel-title'>시스템 로그</h4>", unsafe_allow_html=True)
    log_html = '<div class="log-container">'
    for log in st.session_state.logs[-15:]:
        if '[' in log and ']' in log:
            parts = log.split(']', 1)
            time_part = parts[0] + ']'
            message_part = parts[1] if len(parts) > 1 else ''
            log_html += f'<div class="log-entry"><span class="log-time">{time_part}</span>{message_part}</div>'
        else:
            log_html += f'<div class="log-entry">{log}</div>'
    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)

def render_sidebar(G):
    """사이드바 렌더링"""
    st.markdown("<h4 class='panel-title'>제어 패널</h4>", unsafe_allow_html=True)
    
    # 홀 형태 선택
    venue_type = st.radio(
        "홀 형태",
        ["I형", "T형"],
        index=0 if st.session_state.venue_type == 'I형' else 1,
        horizontal=True
    )
    
    if venue_type != st.session_state.venue_type:
        st.session_state.venue_type = venue_type
        st.session_state.gnn_model_loaded = False  # 모델 재로드 플래그
        add_log(f"홀 형태 변경: {venue_type}")
        st.rerun()


    # 모드 선택
    st.markdown(" ", unsafe_allow_html=True)
    mode = st.radio(
        "모드 선택",
        ["관제 모드", "화재 모드"],
        index=0
    )
    
    # 세션 상태에 현재 모드 저장
    st.session_state.current_mode = mode
    
    color_mode = None
    selected_seat = None
    
    if mode == "화재 모드":
        color_mode, selected_seat = render_fire_mode_controls(G)
    
    # 세션 상태에 color_mode 저장
    st.session_state.current_color_mode = color_mode
    
    st.divider()
    
    # 시스템 로그
    render_system_log()

    return mode, color_mode, selected_seat

def render_fire_mode_controls(G):
    """화재 모드 컨트롤 렌더링"""
    st.markdown("### 화재 시뮬레이션")
    
    # 기존 방식: 셀렉트박스로 시작 노드 선택
    if not st.session_state.simulation_running:
        seat_nodes = sorted([n for n in G.nodes if G.nodes[n]['type'] == 'seat'])
        aisle_nodes = sorted([n for n in G.nodes if G.nodes[n]['type'] == 'aisle'])
        fire_node_select = st.selectbox(
            "시작 노드",
            ["화재 발생 지점 선택"] + seat_nodes + aisle_nodes,
            key="fire_node_select"
        )
        
        if st.button("화재 발생", use_container_width=True, type="primary"):
            if fire_node_select == "화재 발생 지점 선택":
                st.session_state.fire_node = random.choice(seat_nodes)
            else:
                st.session_state.fire_node = fire_node_select
            
            st.session_state.simulation_running = True
            st.session_state.current_time = 0
            st.session_state.show_bottleneck = False  # 초기값: 병목도 OFF

            # people 리스트 초기화
            st.session_state.people = []
            person_id = 0
            
            # evacuation_paths 사용
            if hasattr(st.session_state, 'evacuation_paths'):
                for seat_node, path in st.session_state.evacuation_paths.items():
                    if path:
                        num_people = G.nodes[seat_node].get('capacity', 1)
                        for _ in range(num_people):
                            st.session_state.people.append({
                                'id': person_id,
                                'path': path,
                                'idx': 0,
                                'done': False,
                                'time': 0
                            })
                            person_id += 1
            
            add_log(f"화재 발생 위치: {st.session_state.fire_node}")
            
            # 대피시간 정보 로그 추가
            if hasattr(st.session_state, 'total_evacuation_time') and st.session_state.total_evacuation_time:
                add_log(f"총 대피시간: {st.session_state.total_evacuation_time:.2f}분")
            if hasattr(st.session_state, 'avg_evacuation_time') and st.session_state.avg_evacuation_time:
                add_log(f"평균 대피시간: {st.session_state.avg_evacuation_time:.2f}분")
            
            st.rerun()
    else:
        # 시뮬레이션 실행 중 - 컨트롤과 병목도 토글 표시
        render_simulation_controls()
        
        # 병목도 모니터링 토글
        st.divider()
        show_bottleneck = st.toggle(
            "병목도 모니터링",
            value=st.session_state.get('show_bottleneck', False),
            key="bottleneck_toggle"
        )
        st.session_state.show_bottleneck = show_bottleneck
    
    # 대피 경로 시각화
    selected_seat = render_evacuation_path_selector(G)
    
    # color_mode 결정: 시뮬레이션 중이고 토글 ON일 때만 병목도 표시
    color_mode = None
    if st.session_state.simulation_running and st.session_state.get('show_bottleneck', False):
        color_mode = "병목도"
    
    return color_mode, selected_seat

def render_simulation_controls():
    """시뮬레이션 컨트롤 렌더링"""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⏸️ 정지", use_container_width=True):
            st.session_state.simulation_running = False
            add_log("시뮬레이션 정지됨")
            st.rerun()
    
    with col2:
        if st.button("🔄 리셋", use_container_width=True):
            st.session_state.simulation_running = False
            st.session_state.current_time = 0
            st.session_state.fire_node = None
            st.session_state.evacuation_complete_time = None
            st.session_state.avg_evacuation_time = None
            from data_loader import load_graph_data
            G = load_graph_data()[0]
            for node in G.nodes:
                G.nodes[node]['fire_arrival_time'] = float('inf')
            add_log("시뮬레이션 리셋됨")
            st.rerun()
    
    # 타임라인 슬라이더
    st.markdown("#### 타임라인")
    time_slider = st.slider(
        "경과 시간 (초)",
        0,
        st.session_state.max_time,
        st.session_state.current_time,
        key="time_slider"
    )
    
    if time_slider != st.session_state.current_time:
        st.session_state.current_time = time_slider
        st.rerun()
    
    # 시간 표시
    minutes = st.session_state.current_time // 60
    seconds = st.session_state.current_time % 60
    st.markdown(f"**현재 시간:** {minutes}분 {seconds}초")
    
    # 대피 완료 시간 표시
    if st.session_state.evacuation_complete_time:
        calculate_evacuation_times()
        
        st.markdown("#### 시뮬레이션 결과")
        st.markdown(f"**총 대피완료시간:** {st.session_state.evacuation_complete_time / 60: .1f}분")
        if st.session_state.avg_evacuation_time:
            st.markdown(f"**평균 대피시간:** {st.session_state.avg_evacuation_time / 60:.1f}분")

def calculate_evacuation_times():
    """대피 시간 계산"""
    if hasattr(st.session_state, 'people') and st.session_state.people:
        all_times = [p['time'] for p in st.session_state.people if p.get('done', False)]
        
        if all_times:
            max_time = max(all_times)
            avg_time = sum(all_times) / len(all_times)
            
            st.session_state.evacuation_complete_time = max_time
            st.session_state.avg_evacuation_time = avg_time
        else:
            st.session_state.evacuation_complete_time = st.session_state.max_time
            st.session_state.avg_evacuation_time = st.session_state.max_time
    else:
        st.session_state.evacuation_complete_time = st.session_state.max_time
        st.session_state.avg_evacuation_time = st.session_state.max_time

def render_evacuation_path_selector(G):
    """대피 경로 선택기 렌더링"""
    st.divider()
    st.markdown("### 대피 경로 확인")
    
    seat_nodes = sorted([n for n in G.nodes if G.nodes[n]['type'] == 'seat'])
    selected_seat = st.selectbox("군중 선택", ["선택 안함"] + seat_nodes, key="selected_seat_fire")
    
    if selected_seat != "선택 안함" and st.session_state.evacuation_paths:
        if selected_seat in st.session_state.evacuation_paths:
            path = st.session_state.evacuation_paths[selected_seat]
            if path:
                # 목표 출구
                target_exit = path[-1]
                
                # ========== 출구 상태 확인 (시각화용) ==========
                exit_status_dict = st.session_state.get('exit_status', {})
                is_exit_active = exit_status_dict.get(str(target_exit), True)
                
                # 출구 상태에 따라 다른 색상 표시
                if is_exit_active:
                    st.markdown(f"**목표 출구:** 🟢 {target_exit} (활성)")
                else:
                    st.markdown(f"**목표 출구:** 🔴 {target_exit} (비활성)")
                    # ⭐ 수정: 경고 메시지 변경
                    st.info("출구 상태는 시각화에만 영향을 줍니다.")
                # ==========================================
                
                # 프로토타입 정보 추가 (디버깅용)
                if hasattr(st.session_state, 'pred') and hasattr(st.session_state, 'node_list'):
                    try:
                        node_list = st.session_state.node_list
                        pred = st.session_state.pred
                        seat_idx = node_list.index(selected_seat)
                        proto_id = pred[seat_idx].item()
                        st.markdown(f"**프로토타입:** {proto_id}")
                    except:
                        pass
                
                # 총 거리 계산
                total_distance = 0
                for i in range(len(path) - 1):
                    if G.has_edge(path[i], path[i+1]):
                        total_distance += G[path[i]][path[i+1]]['weight']
                
                # 예상 대피 시간
                avg_speed = 1300
                estimated_time = total_distance / avg_speed

                # 분, 초 변환
                minutes = int(estimated_time // 60)
                seconds = int(estimated_time % 60)

                st.markdown(f"**예상 대피시간:** {minutes}분 {seconds}초")
                
                # 경로 통과 노드 수
                st.markdown(f"**경로 노드 수:** {len(path)}개")

            else:
                st.warning("경로를 찾을 수 없습니다.")
    
    return selected_seat