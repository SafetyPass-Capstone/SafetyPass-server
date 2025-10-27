# replanner.py
"""Fixed-Mapping Dijkstra Replanner - 하이퍼파라미터 기반 동적 경로 재계획"""

import math
import numpy as np
import heapq
from collections import defaultdict, Counter

# ============================================================
# 하이퍼파라미터 (기본값)
# ============================================================
DELTA_T_SEC = 30.0  # 실시간 갱신 주기(초)
NUM_STEPS = 10  # 총 반복 횟수
THETA_IMPROVE = 0.10  # 10% 이상 개선시에만 경로 변경
COOLDOWN_STEPS = 2  # 경로 변경 후 최소 유지 스텝수

ALPHA_FIRE = 0.7  # 화재 위험 가중
BETA_CAP = 0.6  # 용량(작을수록 페널티)
AISLE_BONUS = 0.15  # 통로 노드 선호 (비용↓)
SEAT_PENALTY = 0.10  # 좌석 노드 페널티 (비용↑)
GAMMA_FLOOR = 0.1  # 층 변경 발생 시 고정 패널티
LAMBDA_USAGE = 0.1  # 엣지 사용량 기반 혼잡 페널티

FIRE_CLOSE_THRESH = 0.0  # fire_arrival_time ≤ 이 값이면 폐쇄
SAFETY_MARGIN = 0.0  # 화재 마진(분). 0이면 미사용

# ============================================================
# 유틸리티 함수
# ============================================================
def capacity_of_node(G, n):
    """노드의 용량 반환"""
    return float(G.nodes[n].get('capacity', G.nodes[n].get('exit_capacity', 0.0)) or 0.0)

def edge_capacity(G, u, v):
    """엣지 용량 (양 끝 노드 용량의 min)"""
    return max(min(capacity_of_node(G, u), capacity_of_node(G, v)), 0.0)

def compute_fire_stats(G):
    """화재 도착 시간 통계 계산"""
    arr = []
    for n in sorted(G.nodes()):  # 정렬된 순서로 처리
        ft = G.nodes[n].get('fire_arrival_time', float('inf'))
        if math.isfinite(ft):
            arr.append(float(ft))
    
    if not arr:
        return 1.0, 1.0
    
    arr = np.array(arr, dtype=float)
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    FIRE_MED = max(q50, 1e-6)
    FIRE_IQRs = max((q75 - q25) / 2.0, 1e-3)
    return FIRE_MED, FIRE_IQRs

def update_fire_fields(G, delta_minutes):
    """화재 시간 경과 업데이트 및 폐쇄 노드 체크"""
    # 정렬된 순서로 처리하여 재현 가능성 보장
    for n in sorted(G.nodes()):
        ft = G.nodes[n].get('fire_arrival_time', float('inf'))
        if math.isfinite(ft):
            ft_new = ft - delta_minutes
            G.nodes[n]['fire_arrival_time'] = ft_new
            G.nodes[n]['_closed'] = (ft_new <= FIRE_CLOSE_THRESH)
        else:
            G.nodes[n]['_closed'] = False

# ============================================================
# 가중치 함수 (하이퍼파라미터 반영)
# ============================================================
def make_weight_function(
    G,
    edge_usage=None,
    alpha_fire=ALPHA_FIRE,
    beta_cap=BETA_CAP,
    aisle_bonus=AISLE_BONUS,
    seat_penalty=SEAT_PENALTY,
    gamma_floor=GAMMA_FLOOR,
    lambda_usage=LAMBDA_USAGE,
    safety_margin=SAFETY_MARGIN
):
    """동적 가중치 함수 생성"""
    usage = edge_usage or defaultdict(int)
    
    # 정렬된 순서로 용량 계산
    caps = [capacity_of_node(G, n) for n in sorted(G.nodes())]
    CAP_MEAN = max(np.mean(caps) if len(caps) else 1.0, 1.0)
    FIRE_MED, FIRE_IQRs = compute_fire_stats(G)

    def weight(u, v, edata):
        # 폐쇄 체크
        if G.nodes[u].get('_closed', False) or G.nodes[v].get('_closed', False):
            return float('inf')

        # 기본 베이스 비용
        base = edata.get('weight', edata.get('distance', 1.0))
        if base <= 0:
            base = 1.0

        # 화재 위험 (v 노드에서)
        ft_v = G.nodes[v].get('fire_arrival_time', float('inf'))
        if not math.isfinite(ft_v):
            fire_risk = 0.0
        else:
            eff = ft_v - safety_margin
            fire_risk = 1.0 / (1.0 + math.exp((eff - FIRE_MED) / (FIRE_IQRs + 1e-9)))

        # 용량 패널티 (작을수록 ↑)
        cap_v = capacity_of_node(G, v)
        eff_cap = (cap_v / CAP_MEAN) if cap_v > 0 else 0.0
        cap_pen = 1.0 / (eff_cap + 1e-9)

        # 타입 보정
        t_v = G.nodes[v].get('type')
        type_mul = 1.0
        if t_v == 'aisle':
            type_mul *= (1.0 - aisle_bonus)
        if t_v == 'seat':
            type_mul *= (1.0 + seat_penalty)

        # 층 변경 패널티
        floor_u = G.nodes[u].get('floor')
        floor_v = G.nodes[v].get('floor')
        floor_pen = gamma_floor if (floor_u is not None and floor_v is not None and floor_u != floor_v) else 0.0

        # 혼잡(사용량) 패널티
        ekey = tuple(sorted((u, v)))
        ecap = edge_capacity(G, u, v) or CAP_MEAN
        load_pen = lambda_usage * (usage[ekey] / (ecap + 1e-9))

        mult = (1.0 + alpha_fire * fire_risk + beta_cap * cap_pen)
        return float(base * mult * type_mul + floor_pen + load_pen)
    
    return weight

# ============================================================
# 다익스트라 최단 경로 (단일 목적지)
# ============================================================
def dijkstra_to_target(G, weight, src, tgt):
    """다익스트라 알고리즘으로 src → tgt 최단 경로 계산"""
    if src == tgt:
        return 0.0, [src]
    
    pq = []
    dist = defaultdict(lambda: float('inf'))
    parent = {}
    dist[src] = 0.0
    heapq.heappush(pq, (0.0, src))
    
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == tgt:
            break
        
        # 이웃 노드를 정렬하여 재현 가능성 보장
        neighbors = sorted(G.neighbors(u), key=str)
        for v in neighbors:
            w = weight(u, v, G[u][v])
            if math.isinf(w):
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    
    if dist[tgt] == float('inf'):
        return float('inf'), None
    
    # 경로 복원
    path = [tgt]
    while path[-1] != src:
        path.append(parent[path[-1]])
    path.reverse()
    
    return dist[tgt], path

# ============================================================
# 엣지 사용량 업데이트
# ============================================================
def update_edge_usage_from_paths(paths):
    """경로들로부터 엣지 사용량 계산 (혼잡도 추정)"""
    usage = defaultdict(int)
    # 정렬된 순서로 처리
    for src in sorted(paths.keys(), key=str):
        p = paths[src]
        if not p or len(p) < 2:
            continue
        for i in range(len(p) - 1):
            e = tuple(sorted((p[i], p[i + 1])))
            usage[e] += 1
    return usage

# ============================================================
# 고정 매핑 기반 증분 재계획 루프
# ============================================================
def fixed_mapping_replan_loop(
    G,
    pred_exit,
    num_steps=NUM_STEPS,
    delta_t_sec=DELTA_T_SEC,
    theta_improve=THETA_IMPROVE,
    cooldown_steps=COOLDOWN_STEPS,
    alpha_fire=ALPHA_FIRE,
    beta_cap=BETA_CAP,
    aisle_bonus=AISLE_BONUS,
    seat_penalty=SEAT_PENALTY,
    gamma_floor=GAMMA_FLOOR,
    lambda_usage=LAMBDA_USAGE,
    safety_margin=SAFETY_MARGIN,
    verbose=False
):
    """
    고정 매핑 기반 동적 경로 재계획
    
    Args:
        G: NetworkX 그래프
        pred_exit: {노드: 출구} 매핑
        num_steps: 재계획 반복 횟수
        delta_t_sec: 시간 간격 (초)
        ... (기타 하이퍼파라미터)
    
    Returns:
        initial_paths: step 1의 초기 경로 (스냅샷)
        final_paths: 최종 경로
        final_costs: 최종 비용
    """
    # 초기 스냅샷 저장 공간
    initial_paths, initial_costs = {}, {}
    
    current_paths, current_costs = {}, {}
    cooldown_left = defaultdict(int)
    edge_usage = defaultdict(int)

    # 소스 노드 정렬하여 순서 고정 (seat 노드만)
    sources = sorted([
        n for n in G.nodes 
        if G.nodes[n].get('type') == 'seat'  # seat 노드만!
    ], key=str)
    
    # 유효 출구 없는 소스 제거
    sources = [n for n in sources if pred_exit.get(n) in G.nodes]

    for step in range(1, num_steps + 1):
        if verbose:
            print(f"\n=== [Replan step {step}/{num_steps}] (Δt={delta_t_sec}s) ===")

        # 화재 시간 경과 업데이트 (폐쇄 포함)
        update_fire_fields(G, delta_minutes=(delta_t_sec / 60.0))
        
        # 가중치 함수 갱신 (혼잡 피드백 포함)
        weight = make_weight_function(
            G, edge_usage=edge_usage,
            alpha_fire=alpha_fire, beta_cap=beta_cap,
            aisle_bonus=aisle_bonus, seat_penalty=seat_penalty,
            gamma_floor=gamma_floor, lambda_usage=lambda_usage,
            safety_margin=safety_margin
        )

        new_paths, new_costs = {}, {}
        changed = 0

        # 정렬된 순서로 처리
        for n in sources:
            tgt = pred_exit.get(n, None)
            if (tgt is None) or (G.nodes[tgt].get('_closed', False)):
                # 배정 출구가 폐쇄되었으면 경로 없음
                new_paths[n], new_costs[n] = None, float('inf')
                continue

            cost, path = dijkstra_to_target(G, weight, n, tgt)

            old_cost = current_costs.get(n, float('inf'))
            improve = (old_cost - cost) / (old_cost + 1e-9)

            if cooldown_left[n] > 0 and improve < theta_improve:
                # 유지 (히스테리시스)
                new_paths[n] = current_paths.get(n, path)
                new_costs[n] = current_costs.get(n, cost)
                cooldown_left[n] -= 1
            else:
                # 갱신 (충분히 좋아지거나 초기)
                if improve >= theta_improve or old_cost == float('inf'):
                    changed += 1
                    cooldown_left[n] = cooldown_steps
                new_paths[n], new_costs[n] = path, cost

        # step 1 스냅샷 저장
        if step == 1:
            initial_paths = new_paths.copy()
            initial_costs = new_costs.copy()
            if verbose:
                print("Saved step-1 snapshot")

        current_paths, current_costs = new_paths, new_costs
        edge_usage = update_edge_usage_from_paths(current_paths)

        if verbose:
            print(f"  routed: {len(sources)} | changed: {changed}")
            # 정렬된 순서로 출구 분포 출력
            exit_counter = Counter(pred_exit[n] for n in sources if pred_exit.get(n) is not None)
            exit_dist = dict(sorted(exit_counter.items(), key=lambda x: str(x[0])))
            print(f"  exit distribution: {exit_dist}")

    return initial_paths, current_paths, current_costs