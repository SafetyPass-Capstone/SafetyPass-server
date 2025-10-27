# evacuation.py
"""대피 경로 및 병목도 계산"""

import networkx as nx
from collections import defaultdict
from mapping import (
    all_pairs_shortest_path_lengths,
    map_prototypes_to_exits_capacity_load_iter,
    local_capacity_repair,
    predict_exit_for_nodes
)
from replanner import fixed_mapping_replan_loop
from evacuation_time import compute_evacuation_times_from_paths
import numpy as np
import random
import torch

def generate_evacuation_paths(G, pred, node_list, exit_nodes):
    """
    프로토타입 매핑 + 동적 경로 재계획을 사용한 대피 경로 생성
    
    Args:
        G: NetworkX 그래프
        pred: 프로토타입 ID 예측 (torch.Tensor)
        node_list: 노드 리스트 (정렬되어 있어야 함)
        exit_nodes: 출구 노드 리스트 (정렬되어 있어야 함)
    
    Returns:
        full_recommendation: {좌석_노드: [경로]} 딕셔너리
        pred_exit: {노드: 출구} 매핑
        apsp: 모든 쌍 최단 거리
        total_time: 총 대피 시간
        mean_time: 평균 대피 시간
    """
    # 랜덤 시드 고정
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    
    # deterministic 설정
    if torch.cuda.is_available():
        torch.cuda.manual_seed(7)
        torch.cuda.manual_seed_all(7)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 프로토타입 ID를 numpy로 변환
    proto_id = pred.cpu().numpy()
    
    # APSP 계산
    apsp = all_pairs_shortest_path_lengths(G)
    
    # Hungarian 알고리즘으로 프로토타입 → 출구 매핑
    proto_to_exit = map_prototypes_to_exits_capacity_load_iter(
        G, node_list, proto_id, exit_nodes, apsp,
        iters=3, cap_key='capacity', alpha=1.6, lambda_load=0.5
    )
    
    # 각 노드의 출구 예측
    pred_exit = predict_exit_for_nodes(proto_id, proto_to_exit, node_list)
    
    # Local Repair (용량 제약 고려)
    # seat_nodes는 이미 정렬된 node_list에서 필터링하므로 순서 유지됨
    seat_nodes = [n for n in node_list if G.nodes[n].get('type') == 'seat']
    
    # exit_nodes도 정렬된 순서대로 처리
    exit_caps_arr = np.array([
        float(G.nodes[e].get('capacity', G.nodes[e].get('exit_capacity', 1.0))) 
        for e in exit_nodes
    ], float)
    
    improved = local_capacity_repair(G, seat_nodes, exit_nodes, pred_exit, apsp, exit_caps_arr, slack=0.15)
    
    print(f"[Local repair] seats improved: {improved}")
    print("Prototype → Exit mapping:")
    # 정렬된 순서로 출력
    for k in sorted(proto_to_exit.keys()):
        print(f"  proto {k} → {proto_to_exit[k]}")
    
    # Fixed-Mapping Dijkstra Replanner로 동적 경로 계획
    initial_paths, final_paths, final_costs = fixed_mapping_replan_loop(
        G, pred_exit,
        num_steps=1,  # 초기 경로만 계산
        delta_t_sec=30.0,
        theta_improve=0.10,
        cooldown_steps=2,
        verbose=True
    )
    
    # 좌석 노드만 필터링하여 반환 (순서 유지)
    full_recommendation = {}
    for seat_node in seat_nodes:
        full_recommendation[seat_node] = initial_paths.get(seat_node, None)
    
    # 대피 시간 계산
    print("\n" + "="*50)
    print("📊 대피 시간 계산")
    print("="*50)
    total_time, mean_time, seat_times = compute_evacuation_times_from_paths(
        G, full_recommendation, speed_mps=1.3
    )
    
    return full_recommendation, pred_exit, apsp, total_time, mean_time

from collections import defaultdict

def calculate_bottleneck(G, full_recommendation):
    """
    병목도 계산:
    - 각 노드별로 '그 노드를 지나가려는 총 인원(demand)'
    - 노드의 용량(capacity 또는 처리량)을 나눠서 병목도를 계산
    - 병목도 = demand / capacity
    """
    node_demand = defaultdict(float)

    # 전체 인원(진짜 사람 수)도 계산
    total_people = 0.0

    # 좌석 노드를 정렬된 순서로 처리 (재현성 유지)
    for seat_node in sorted(full_recommendation.keys(), key=str):
        path = full_recommendation[seat_node]

        if not path:
            continue

        # 이 좌석(seat_node)에 있는 사람 수
        seat_capacity = G.nodes[seat_node].get('capacity', 1.0)
        total_people += seat_capacity

        # 이 좌석 사람들이 path 전체를 따라 이동한다고 가정
        for node in path:
            node_demand[node] += seat_capacity

    # 이제 각 노드별 병목도 계산
    bottleneck_map = {}

    for node in G.nodes():
        # 이 노드의 "처리 용량" (있으면 쓰고, 없으면 capacity / 없으면 1.0)
        node_capacity = (
            G.nodes[node].get('capacity_per_tick', None)  # 예: 초당/틱당 통과 가능 인원
            or G.nodes[node].get('capacity', None)        # 좌석/통로 수용 인원
            or 1.0
        )

        demand = node_demand.get(node, 0.0)

        # 병목도: 수요 / 용량
        bottleneck_value = demand / node_capacity if node_capacity > 0 else float('inf')

        bottleneck_map[node] = bottleneck_value

        # 그래프에도 저장
        G.nodes[node]['bottleneck'] = bottleneck_value
        G.nodes[node]['demand_people'] = demand
        G.nodes[node]['node_capacity_used_for_bottleneck'] = node_capacity

    # total_people은 전체 인구수 리포트 용으로 같이 반환하면 유용함
    return bottleneck_map, total_people
