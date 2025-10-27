# evacuation_time.py
"""대피 시간 계산 모듈"""

import math
import numpy as np
from collections import Counter
'''
def compute_evacuation_times_minutes(G, seat_nodes, pred_exit, apsp, speed_mps=1.3):
    """
    총 대피시간, 평균 대피시간, 노드별 대피시간(분 단위)을 계산
    
    Args:
        G: NetworkX 그래프
        seat_nodes: 좌석 노드 리스트
        pred_exit: 각 좌석 노드 -> 출구 노드 매핑 딕셔너리
        apsp: all_pairs_shortest_path_lengths(G)
        speed_mps: 이동 속도 (m/s), 기본 = 1.3
    
    Returns:
        total_evac_time_min: 전체 대피 완료 시간 (분)
        mean_evac_time_min: 인원 수 기준 평균 대피시간 (분)
        seat_times_min: 각 좌석 노드별 대피시간(분) 딕셔너리
    """
    seat_times_min = {}
    total_people = 0
    total_weighted_time = 0.0

    for n in seat_nodes:
        e = pred_exit.get(n)
        if e is None or e not in apsp.get(n, {}):
            continue

        dist_m = apsp[n][e] / 1000.0          # 거리(m) - mm에서 변환
        evac_time_s = dist_m / speed_mps      # 초 단위
        evac_time_min = evac_time_s / 60.0    # 분 단위 변환

        seat_times_min[n] = evac_time_min

        cap = float(G.nodes[n].get('capacity', 1.0))
        total_people += cap
        total_weighted_time += evac_time_min * cap

    # 전체 대피 완료 시간 (가장 오래 걸린 노드 기준)
    total_evac_time_min = max(seat_times_min.values()) if seat_times_min else float('nan')
    
    # 인원 기준 평균 대피 시간
    mean_evac_time_min = total_weighted_time / max(total_people, 1.0)

    print("💨 총 인원:", int(total_people))
    print(f"⏱️ 평균 대피시간 (인원 가중): {mean_evac_time_min:.2f}분")
    print(f"🏁 전체 대피 완료시간 (최대값): {total_evac_time_min:.2f}분")

    return total_evac_time_min, mean_evac_time_min, seat_times_min
'''
def compute_evacuation_times_from_paths(G, full_recommendation, speed_mps=1.3):
    """
    경로 딕셔너리로부터 대피 시간 계산
    
    Args:
        G: NetworkX 그래프
        full_recommendation: {좌석_노드: [경로]} 딕셔너리
        speed_mps: 이동 속도 (m/s), 기본 = 1.3
    
    Returns:
        total_evac_time_min: 전체 대피 완료 시간 (분)
        mean_evac_time_min: 인원 수 기준 평균 대피시간 (분)
        seat_times_min: 각 좌석 노드별 대피시간(분) 딕셔너리
    """
    seat_times_min = {}
    total_people = 0
    total_weighted_time = 0.0

    for seat_node, path in full_recommendation.items():
        if not path or len(path) < 2:
            continue

        # 경로의 총 거리 계산
        total_distance = 0.0
        for i in range(len(path) - 1):
            if G.has_edge(path[i], path[i + 1]):
                total_distance += G[path[i]][path[i + 1]].get('weight', 0)

        dist_m = total_distance / 1000.0      # 거리(m) - mm에서 변환
        evac_time_s = dist_m / speed_mps      # 초 단위
        evac_time_min = evac_time_s / 60.0    # 분 단위 변환

        seat_times_min[seat_node] = evac_time_min

        cap = float(G.nodes[seat_node].get('capacity', 1.0))
        total_people += cap
        total_weighted_time += evac_time_min * cap

    # 전체 대피 완료 시간 (가장 오래 걸린 노드 기준)
    total_evac_time_min = max(seat_times_min.values()) if seat_times_min else float('nan')
    
    # 인원 기준 평균 대피 시간
    mean_evac_time_min = total_weighted_time / max(total_people, 1.0)

    return total_evac_time_min, mean_evac_time_min, seat_times_min