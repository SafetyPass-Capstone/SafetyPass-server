# fire_simulation.py
"""화재 시뮬레이션 관련 함수"""

import networkx as nx
from config import FIRE_SPREAD_SPEED, FIRE_BUFFER_MARGIN

def calculate_fire_spread(G, fire_node, current_time_min):
    """화재 확산 계산"""
    for node in G.nodes:
        if G.nodes[node]['fire_arrival_time'] == float('inf'):
            try:
                distance = nx.shortest_path_length(G, fire_node, node, weight='weight')
                arrival_time = (distance / FIRE_SPREAD_SPEED) + FIRE_BUFFER_MARGIN
                
                if arrival_time <= current_time_min:
                    G.nodes[node]['fire_arrival_time'] = arrival_time
            except nx.NetworkXNoPath:
                continue

def get_fire_nodes(G, current_time_min):
    """화재 도달 노드 및 접근 중 노드 반환"""
    fire_nodes = []
    fire_approaching = []
    
    for node in G.nodes:
        if G.nodes[node]['type'] in ['seat', 'aisle']:
            fire_time = G.nodes[node].get('fire_arrival_time', float('inf'))
            
            if fire_time <= current_time_min:
                fire_nodes.append(node)
            elif fire_time <= current_time_min + 1:
                fire_approaching.append(node)
    
    return fire_nodes, fire_approaching