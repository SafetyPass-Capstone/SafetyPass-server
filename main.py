"""메인 실행 파일 - Redis 연동 추가"""

import random
import numpy as np
import torch

# 랜덤 시드 고정
RANDOM_SEED = 7
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 모듈 임포트
import streamlit as st
import time
from theme import DARK_THEME_CSS
from session_state import initialize_session_state
from data_loader import load_graph_data, load_seat_units_csv
from model import load_trained_model
from evacuation import generate_evacuation_paths, calculate_bottleneck
from fire_simulation import calculate_fire_spread
from visualization import create_graph_figure
from ui_components import render_system_status, render_exit_controls
from sidebar import render_sidebar
from simulation_logic import run_simulation_step, check_simulation_complete
from utils import add_log


# Redis 연동 추가
from redis_manager import get_redis_manager
import uuid

# 페이지 설정
st.set_page_config(
    page_title="대피 시뮬레이션 관제 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 다크 테마 CSS 적용
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

# 세션 상태 초기화
initialize_session_state()

# ========== Redis 연동 초기화 ==========
# 세션 ID를 고정값으로 설정 (모든 사용자가 공유)
SHARED_SESSION_ID = "evacuation-system-shared"

if 'session_id' not in st.session_state:
    st.session_state.session_id = SHARED_SESSION_ID
    add_log(f"공유 세션 사용: {SHARED_SESSION_ID}")

# Redis 매니저 초기화
redis_mgr = get_redis_manager()

def sync_state_to_redis():
    """중요 상태를 Redis에 동기화"""
    if not redis_mgr.is_connected():
        return
    
    session_id = st.session_state.session_id
    
    # 시뮬레이션 상태
    redis_mgr.save_state(session_id, 'simulation_running', st.session_state.get('simulation_running', False))
    redis_mgr.save_state(session_id, 'current_time', st.session_state.get('current_time', 0))
    redis_mgr.save_state(session_id, 'fire_node', st.session_state.get('fire_node', None))
    redis_mgr.save_state(session_id, 'venue_type', st.session_state.get('venue_type', 'I'))
    
    # ========== 그래프 엣지 가중치 저장 추가 ==========
    if 'graph_edges' not in st.session_state:
        # 그래프 로드
        _G = load_graph_data(venue_type=st.session_state.venue_type)
        G_temp = _G[0] if isinstance(_G, tuple) else _G
        
        if G_temp is not None:
            # 그래프의 모든 엣지 가중치를 딕셔너리로 변환
            edge_weights = {}
            for u, v, data in G_temp.edges(data=True):
                weight = data.get('weight', 0)
                # 양방향 저장 (문자열 키 사용)
                edge_weights[f"{u},{v}"] = float(weight)
                edge_weights[f"{v},{u}"] = float(weight)
            
            st.session_state.graph_edges = edge_weights
            redis_mgr.save_state(session_id, 'graph_edges', edge_weights)
            add_log(f"그래프 엣지 저장 완료: {len(edge_weights)}개")
    else:
        # 이미 저장된 경우 Redis 업데이트
        redis_mgr.save_state(session_id, 'graph_edges', st.session_state.graph_edges)
    
    # apsp 저장 (중요!)
    if 'apsp' in st.session_state and st.session_state.apsp:
        redis_mgr.save_state(session_id, 'apsp', st.session_state.apsp)
        add_log(f"APSP 저장 완료: {len(st.session_state.apsp)}개 경로")
    
    # evacuation_paths 저장
    if 'evacuation_paths' in st.session_state:
        redis_mgr.save_state(session_id, 'evacuation_paths', st.session_state.evacuation_paths)
    
    if 'exit_status' in st.session_state:
        redis_mgr.save_state(session_id, 'exit_status', st.session_state.exit_status)
    
    # 대피 데이터
    if st.session_state.get('gnn_model_loaded'):
        redis_mgr.save_state(session_id, 'total_evacuation_time', st.session_state.get('total_evacuation_time'))
        redis_mgr.save_state(session_id, 'avg_evacuation_time', st.session_state.get('avg_evacuation_time'))
    
    # ========== 화재 확산 데이터 동기화 추가 ==========
    if st.session_state.get('simulation_running') and st.session_state.get('fire_node'):
        from fire_simulation import get_fire_nodes
        
        current_time_min = st.session_state.get('current_time', 0) / 60
        fire_nodes, fire_approaching = get_fire_nodes(G, current_time_min)
        
        # 각 노드별 화재 도달 시간
        fire_arrival_times = {}
        for node in G.nodes:
            fire_time = G.nodes[node].get('fire_arrival_time', float('inf'))
            if fire_time != float('inf'):
                fire_arrival_times[str(node)] = fire_time
        
        fire_data = {
            "fire_origin": st.session_state.fire_node,
            "current_time_seconds": st.session_state.get('current_time', 0),
            "current_time_minutes": current_time_min,
            "fire_reached_nodes": list(fire_nodes),
            "fire_approaching_nodes": list(fire_approaching),
            "fire_arrival_times": fire_arrival_times,
            "total_affected_nodes": len(fire_nodes) + len(fire_approaching)
        }
        
        redis_mgr.save_state(session_id, 'fire_spread_data', fire_data)
        add_log(f"화재 확산 데이터 동기화: 도달 {len(fire_nodes)}개, 접근 {len(fire_approaching)}개")
    else:
        # 화재가 없으면 데이터 초기화
        redis_mgr.save_state(session_id, 'fire_spread_data', None)
    # ==============================================

def sync_fire_spread_to_redis():
    """화재 확산 데이터를 Redis에 동기화"""
    if not redis_mgr.is_connected():
        return
    
    if st.session_state.simulation_running and st.session_state.fire_node:
        from fire_simulation import get_fire_nodes
        
        current_time_min = st.session_state.current_time / 60
        fire_nodes, fire_approaching = get_fire_nodes(G, current_time_min)
        
        fire_data = {
            "fire_origin": st.session_state.fire_node,
            "current_time": st.session_state.current_time,
            "fire_reached": list(fire_nodes),  # 화재 도달 노드
            "fire_approaching": list(fire_approaching),  # 화재 접근 노드
            "fire_node_count": len(fire_nodes)
        }
        
        redis_mgr.save_state(
            st.session_state.session_id, 
            "fire_spread_data", 
            fire_data
        )

def check_redis_commands():
    """Redis에서 외부 명령 확인 및 처리"""
    if not redis_mgr.is_connected():
        return
    
    session_id = st.session_state.session_id
    command = redis_mgr.get_command(session_id)
    
    if command:
        action = command.get('action')
        add_log(f"외부 명령 수신: {action}")
        
        if action == 'start':
            st.session_state.fire_node = command.get('fire_node')
            venue_type = command.get('venue_type', 'I')
            if st.session_state.venue_type != venue_type:
                st.session_state.venue_type = venue_type
                st.session_state.gnn_model_loaded = False
            st.session_state.simulation_running = True
            st.session_state.current_time = 0
            add_log(f"시뮬레이션 시작: 화재노드={st.session_state.fire_node}")
        
        elif action == 'stop':
            st.session_state.simulation_running = False
            add_log("시뮬레이션 중지")
        
        elif action == 'reset':
            st.session_state.simulation_running = False
            st.session_state.current_time = 0
            st.session_state.fire_node = None
            add_log("시뮬레이션 리셋")

# venue_type 변경 시 모델 캐시 클리어 (main.py에서 처리)
if st.session_state.get('prev_venue_type_for_model') != st.session_state.venue_type:
    from model import load_trained_model
    load_trained_model.clear()
    st.session_state.prev_venue_type_for_model = st.session_state.venue_type
    # graph_edges도 클리어하여 재생성되도록
    if 'graph_edges' in st.session_state:
        del st.session_state['graph_edges']
    add_log(f"모델 캐시 클리어: {st.session_state.venue_type}")

# 클릭 선택 후보 기본값 보장
if 'fire_node_candidate' not in st.session_state:
    st.session_state.fire_node_candidate = None

# 그래프 로드
_G = load_graph_data(venue_type=st.session_state.venue_type)

# 튜플 체크 및 언팩
if isinstance(_G, tuple):
    G = _G[0]
else:
    G = _G

if G is None:
    st.error("그래프를 로드할 수 없습니다.")
    st.stop()

# GNN 모델 로드 및 대피 경로 생성
if not st.session_state.gnn_model_loaded:
    with st.spinner("GNN 모델 로딩 중..."):
        result = load_trained_model(G, st.session_state.venue_type)
        
        if result[0] is not None:
            encoder, proto_head, exit_mapper, pyg_data, node_list, exit_nodes, pred = result
            
            # ========== 출구 상태 초기화 (모두 활성화) ==========
            if 'exit_status' not in st.session_state:
                st.session_state.exit_status = {str(n): True for n in exit_nodes}
            
            # ⭐ 수정: 항상 전체 출구로 경로 생성 (고정)
            add_log(f"전체 {len(exit_nodes)}개 출구로 경로 생성")
            # ==========================================
            
            # 대피 경로 생성 (전체 출구 사용)
            from evacuation import generate_evacuation_paths
            evacuation_result = generate_evacuation_paths(
                G, pred, node_list, exit_nodes  # ← 전체 출구 사용 (고정)
            )
            
            st.session_state.evacuation_paths = evacuation_result[0]
            st.session_state.pred_exit = evacuation_result[1]
            st.session_state.apsp = evacuation_result[2]
            st.session_state.total_evacuation_time = evacuation_result[3]
            st.session_state.avg_evacuation_time = evacuation_result[4]
            
            # 병목도 계산
            bottleneck_map, total_people = calculate_bottleneck(
                G,
                st.session_state.evacuation_paths
            )
            st.session_state.bottleneck_data = bottleneck_map
            
            st.session_state.gnn_model_loaded = True
            st.session_state.encoder = encoder
            st.session_state.proto_head = proto_head
            st.session_state.exit_mapper = exit_mapper
            st.session_state.exit_nodes = exit_nodes  # 전체 출구 저장
            st.session_state.node_list = node_list
            st.session_state.pred = pred
            
            sync_state_to_redis()
            add_log("초기 상태 Redis 동기화 완료")
        
        else:
            st.error("모델 로드에 실패했습니다.")
            st.stop()

# ========== 외부 명령 확인 ==========
check_redis_commands()
sync_state_to_redis()


# 메인 화면 (좌정렬)
st.markdown("<h1 class='main-header'>대피 시뮬레이션 관제 시스템</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Evacuation Simulation Control System v2.1</p>", unsafe_allow_html=True)

# Redis 연결 상태 표시 (디버그용 - 옵션)
if redis_mgr.is_connected():
    st.sidebar.success(f"🟢 Redis 연결됨 (Session: {st.session_state.session_id[:8]}...)")
else:
    st.sidebar.warning("🔴 Redis 연결 끊김 (로컬 모드)")

# 3단 레이아웃: 제어 패널(좌) | 그래프(중앙) | 시스템 상태(우)
col1, col2, col3 = st.columns([1, 2.5, 1])

with col1:
    # 제어 패널 (기존 사이드바)
    mode, color_mode, selected_seat = render_sidebar(G)

with col3:
    # 시스템 상태 패널
    render_system_status(G)
    render_exit_controls(G)
    
with col2:
    # 화재 확산 계산
    if mode == "화재 모드" and st.session_state.simulation_running and st.session_state.fire_node:
        fire_node = st.session_state.fire_node
        current_time_min = st.session_state.current_time / 60
        calculate_fire_spread(G, fire_node, current_time_min)
    
    # 그래프 시각화
    if 'seat_units_df' not in st.session_state or st.session_state.seat_units_df is None:
        st.session_state.seat_units_df = load_seat_units_csv(st.session_state.venue_type)
    fig = create_graph_figure(G, mode, color_mode, st.session_state, venue_type=st.session_state.venue_type)
    st.plotly_chart(fig, use_container_width=True)

# ========== Redis 상태 동기화 ==========
# 시뮬레이션 실행 중이면 상태를 Redis에 주기적으로 동기화
if st.session_state.simulation_running:
    sync_state_to_redis()

# 자동 새로고침 (화재 모드 실행 중)
if mode == "화재 모드" and st.session_state.simulation_running:
    time.sleep(1)
    
    # 시뮬레이션 로직 실행
    run_simulation_step()
    
    # Redis 동기화
    sync_state_to_redis()
    
    # 시뮬레이션 완료 체크
    if not check_simulation_complete():
        st.rerun()
    else:
        # 완료 시에도 최종 상태 동기화
        sync_state_to_redis()