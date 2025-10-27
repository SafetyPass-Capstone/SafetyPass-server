# mapping.py
"""프로토타입-출구 매핑 모듈 (Hungarian Algorithm + Local Repair)"""

import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from collections import Counter

def all_pairs_shortest_path_lengths(G):
    """모든 노드 쌍 간의 최단 거리 계산"""
    if G.number_of_edges() == 0:
        return {}
    return dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

def map_prototypes_to_exits_capacity_load_iter(
    G, node_list, proto_id, exits, apsp, iters=3,
    cap_key='capacity', eps=1e-8, alpha=1.6, lambda_load=0.5
):
    """
    Hungarian 알고리즘 + 용량 균형을 고려한 프로토타입-출구 매핑
    
    Args:
        G: NetworkX 그래프
        node_list: 노드 리스트 (정렬된 순서 유지)
        proto_id: 각 노드의 프로토타입 ID (numpy array)
        exits: 출구 노드 리스트 (정렬된 순서)
        apsp: 모든 노드 쌍 최단 거리 딕셔너리
        iters: 반복 횟수
        cap_key: 용량 키
        alpha: 용량 가중치 지수
        lambda_load: 부하 패널티 가중치
    
    Returns:
        proto_to_exit: {프로토타입_ID: 출구_노드} 매핑
    """
    if len(exits) == 0:
        return {int(k): None for k in range(int(proto_id.max() + 1))}
    
    # 출구 용량 정규화
    raw_caps = np.array([
        float(G.nodes[e].get(cap_key, G.nodes[e].get('exit_capacity', 1.0))) 
        for e in exits
    ], dtype=float)
    eff_caps = raw_caps / max(raw_caps.mean(), eps)
    
    # 프로토타입별 노드 그룹화 (순서 유지)
    K = int(proto_id.max() + 1)
    proto_nodes = []
    for k in range(K):
        # 정렬된 node_list 순서대로 필터링하므로 순서 유지됨
        idxs = [
            i for i, p in enumerate(proto_id)
            if p == k and G.nodes[node_list[i]].get('type') in ('seat', 'aisle')
        ]
        proto_nodes.append([node_list[i] for i in idxs])
    
    # 기본 비용 행렬 계산 (거리 / 용량)
    base_cost = np.zeros((K, len(exits)), dtype=float)
    for k in range(K):
        nodes_k = proto_nodes[k]
        if not nodes_k:
            base_cost[k, :] = 1e9
            continue
        for j, e in enumerate(exits):
            # 정렬된 순서로 거리 계산
            ds = [apsp.get(n, {}).get(e, 1e9) for n in nodes_k]
            avg_d = np.mean(ds) if len(ds) > 0 else 1e9
            base_cost[k, j] = avg_d / (eff_caps[j] ** alpha + eps)
    
    # 반복적 매핑 (부하 균형 고려)
    current_mapping = {}
    current_load = Counter()
    cost = base_cost.copy()
    
    for _ in range(iters):
        # 현재 부하에 따라 비용 조정
        for j, e in enumerate(exits):
            load_pen = lambda_load * (current_load[e] / (eff_caps[j] + eps))
            cost[:, j] = base_cost[:, j] + load_pen
        
        # Hungarian 알고리즘으로 최적 매핑
        r, c = linear_sum_assignment(cost)
        current_mapping = {int(rk): exits[int(ck)] for rk, ck in zip(r, c)}
        
        # 현재 매핑에 따른 부하 재계산
        current_load = Counter()
        for k in range(K):
            if k in current_mapping:
                e = current_mapping[k]
                seat_nodes_k = [n for n in proto_nodes[k] if G.nodes[n].get('type') == 'seat']
                current_load[e] += sum(float(G.nodes[n].get('capacity', 1.0)) for n in seat_nodes_k)
    
    # 매핑되지 않은 프로토타입 처리 (deterministic하게)
    for k in range(K):
        if k not in current_mapping:
            j = int(np.argmin(cost[k]))
            current_mapping[k] = exits[j]
    
    return current_mapping

def local_capacity_repair(G, seat_nodes, exits, pred_exit, apsp, exit_caps, slack=0.15):
    """
    용량 제약을 고려한 지역 수리
    
    Args:
        G: NetworkX 그래프
        seat_nodes: 좌석 노드 리스트 (정렬된 순서)
        exits: 출구 노드 리스트 (정렬된 순서)
        pred_exit: 현재 예측 {노드: 출구}
        apsp: 모든 노드 쌍 최단 거리
        exit_caps: 출구별 용량 배열
        slack: 허용 오차 (15%)
    
    Returns:
        improved: 개선된 좌석 수
    """
    # 현재 출구별 사용량 계산
    exit_use = Counter()
    for n in seat_nodes:
        e = pred_exit.get(n, None)
        if e in exits:
            exit_use[e] += 1
    
    # 출구별 용량 제한
    cap_limit = {e: float(exit_caps[i]) for i, e in enumerate(exits)}
    
    def dist(n, e):
        return apsp.get(n, {}).get(e, float('inf'))
    
    improved = 0
    # 정렬된 순서로 처리하여 재현 가능성 보장
    for n in seat_nodes:
        e_cur = pred_exit.get(n, None)
        d_cur = dist(n, e_cur) if e_cur in exits else float('inf')
        
        # 모든 출구까지의 거리 정렬 (거리 동일 시 출구 이름 순으로 정렬)
        d_list = sorted(
            [(e, dist(n, e)) for e in exits], 
            key=lambda x: (x[1], str(x[0]))  # 거리, 출구명 순으로 정렬
        )
        if not d_list:
            continue
        
        e_best, d_best = d_list[0]
        
        # 현재 할당이 최적에 가까우면 스킵
        if np.isfinite(d_cur) and d_cur <= d_best * (1.0 + slack):
            continue
        
        # 용량 여유가 있는 더 가까운 출구로 재할당
        for e_new, d_new in d_list:
            if exit_use[e_new] < cap_limit.get(e_new, float('inf')):
                if e_cur in exits:
                    exit_use[e_cur] -= 1
                exit_use[e_new] += 1
                pred_exit[n] = e_new
                improved += 1
                break
    
    return improved

def predict_exit_for_nodes(proto_id, proto_to_exit, node_list):
    """
    프로토타입 ID와 매핑을 사용하여 각 노드의 출구 예측
    
    Args:
        proto_id: 각 노드의 프로토타입 ID (numpy array)
        proto_to_exit: {프로토타입_ID: 출구_노드} 매핑
        node_list: 노드 리스트 (정렬된 순서)
    
    Returns:
        pred: {노드: 출구} 딕셔너리
    """
    pred = {}
    # 정렬된 node_list 순서대로 처리
    for i, node in enumerate(node_list):
        p = int(proto_id[i])
        pred[node] = proto_to_exit.get(p, None)
    return pred