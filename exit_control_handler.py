"""출구 제어 및 경로 재계산 핸들러"""
import streamlit as st
from evacuation import generate_evacuation_paths, calculate_bottleneck
from utils import add_log


def get_active_exits(G, exit_status_dict):
    """
    활성화된 출구만 필터링 (정렬된 순서로 반환)
    
    Args:
        G: NetworkX 그래프
        exit_status_dict: {exit_name: True/False} 딕셔너리
    
    Returns:
        active_exits: 활성화된 출구 노드 리스트 (정렬됨)
    """
    all_exits = sorted([n for n in G.nodes if G.nodes[n].get('type') == 'exit'], key=str)
    
    active_exits = []
    for exit_node in all_exits:
        exit_name = str(exit_node)
        # 상태 확인 (기본값 True)
        is_active = exit_status_dict.get(exit_name, True)
        if is_active:
            active_exits.append(exit_node)
    
    # 정렬된 순서 보장
    return sorted(active_exits, key=str)


def recalculate_paths_with_active_exits(G):
    """
    활성화된 출구만 사용하여 대피 경로 재계산
    
    Args:
        G: NetworkX 그래프
    
    Returns:
        bool: 성공 여부
    """
    try:
        # 출구 상태 가져오기
        exit_status_dict = st.session_state.get('exit_status', {})
        
        # 활성화된 출구만 필터링
        active_exits = get_active_exits(G, exit_status_dict)
        
        # ========== 디버그 로그 추가 ==========
        all_exits = [n for n in G.nodes if G.nodes[n].get('type') == 'exit']
        st.sidebar.write("### 🔍 디버그 정보")
        st.sidebar.write(f"전체 출구: {sorted(all_exits, key=str)}")
        st.sidebar.write(f"활성 출구: {sorted(active_exits, key=str)}")
        st.sidebar.write(f"출구 상태: {exit_status_dict}")
        # ====================================
        
        if len(active_exits) == 0:
            st.error("⚠️ 최소 1개 이상의 출구가 활성화되어야 합니다!")
            add_log("경로 재계산 실패: 활성화된 출구 없음")
            return False
        
        # 출구를 정렬하여 일관성 보장
        active_exits = sorted(active_exits, key=str)
        add_log(f"활성 출구: {len(active_exits)}개 - {active_exits}")
        
        # 기존 모델 데이터 사용
        if not st.session_state.get('gnn_model_loaded'):
            st.warning("모델이 로드되지 않았습니다.")
            return False
        
        pred = st.session_state.pred
        node_list = st.session_state.node_list
        
        # 대피 경로 재생성 (활성 출구만 사용)
        with st.spinner(f"대피 경로 재계산 중 (활성 출구: {len(active_exits)}개)..."):
            # ========== 디버깅 강화 ==========
            st.sidebar.write("### ⚙️ 경로 재계산 진행 중")
            st.sidebar.write(f"활성 출구: {sorted(active_exits, key=str)}")
            # ============================
            
            evacuation_result = generate_evacuation_paths(
                G, pred, node_list, active_exits  # 활성 출구만 전달
            )
            
            st.session_state.evacuation_paths = evacuation_result[0]
            st.session_state.pred_exit = evacuation_result[1]
            st.session_state.apsp = evacuation_result[2]
            st.session_state.total_evacuation_time = evacuation_result[3]
            st.session_state.avg_evacuation_time = evacuation_result[4]
            
            # ========== 재계산 결과 확인 ==========
            st.sidebar.write("### ✅ 재계산 완료")
            exit_distribution = {}
            for seat, path in st.session_state.evacuation_paths.items():
                if path and len(path) > 0:
                    target_exit = str(path[-1])
                    exit_distribution[target_exit] = exit_distribution.get(target_exit, 0) + 1
            
            st.sidebar.write("출구별 배정:")
            for exit_name in sorted(exit_distribution.keys()):
                st.sidebar.write(f"  {exit_name}: {exit_distribution[exit_name]}명")
            # ===================================
            
            # 병목도 재계산
            bottleneck_map, total_people = calculate_bottleneck(
                G,
                st.session_state.evacuation_paths
            )
            st.session_state.bottleneck_data = bottleneck_map
            
            add_log(f"✅ 경로 재계산 완료 - 평균 대피 시간: {evacuation_result[4]:.1f}초")
            return True
    
    except Exception as e:
        st.error(f"경로 재계산 오류: {e}")
        add_log(f"경로 재계산 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def handle_exit_toggle_change(G, exit_name, new_status):
    """
    출구 토글 변경 처리
    
    Args:
        G: NetworkX 그래프
        exit_name: 출구 이름
        new_status: 새로운 상태 (True/False)
    """
    # 상태 업데이트
    if 'exit_status' not in st.session_state:
        st.session_state.exit_status = {}
    
    st.session_state.exit_status[exit_name] = new_status
    
    status_text = "활성화" if new_status else "비활성화"
    add_log(f"출구 {exit_name} {status_text}")
    
    # ========== 수정: 위젯 키 직접 수정 제거 ==========
    # 위젯 키는 직접 수정할 수 없으므로 제거
    # 사용자가 수동으로 다시 선택하도록 함
    # ==============================================
    
    # 경로 재계산
    recalculate_paths_with_active_exits(G)


def validate_exit_status(G):
    """
    출구 상태 검증 - 최소 1개는 활성화되어야 함
    
    Returns:
        bool: 유효한 상태인지 여부
    """
    exit_status_dict = st.session_state.get('exit_status', {})
    active_exits = get_active_exits(G, exit_status_dict)
    
    return len(active_exits) > 0