# data_loader.py
"""데이터 로딩 모듈"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import random
import math

@st.cache_data
def load_graph_data(venue_type='I형'):
    """CSV 파일에서 그래프 데이터 로드 (홀 형태별)"""
    
    # ========== venue_type 정규화 추가 ==========
    # 다양한 입력값을 표준 형태로 통일
    venue_type_str = str(venue_type).strip().upper()
    
    if venue_type_str in ['I', 'I형', 'INDOOR']:
        venue_type = 'I형'
        suffix = 'I'
    elif venue_type_str in ['T', 'T형', 'TRAFFIC']:
        venue_type = 'T형'
        suffix = 'T'
    else:
        # 기본값
        venue_type = 'I형'
        suffix = 'I'
    
    print(f"[load_graph_data] Normalized: '{venue_type}' → suffix='{suffix}'")
    # ==========================================
    
    # 랜덤 시드 고정 (그래프 생성 시)
    random.seed(7)
    np.random.seed(7)
    
    try:
        # CSV 파일 로드 (suffix 사용)
        seats_df = pd.read_csv(f"data/SeatNodes_{suffix}.csv")
        aisle_df = pd.read_csv(f"data/AisleNodes_{suffix}.csv")
        exit_df = pd.read_csv(f"data/ExitNodes_{suffix}.csv")
        edges_df = pd.read_csv(f"data/Edges_{suffix}_withXY.csv")
        
        G = nx.Graph()
        crowd_types = [0, 1, 2]
        
        # 좌석 노드 추가 (deterministic하게)
        for idx, row in seats_df.iterrows():
            # 노드별로 고정된 시드 사용
            node_seed = hash(row['node']) % (2**32)
            local_rng = np.random.RandomState(node_seed)
            
            # crowd_type = local_rng.choice(crowd_types)  # 랜덤 대신 규칙 적용
            if int(row['floor']) == 0 or int(row['node_count']) >= 60:
                crowd_type = 2  # 0층이거나 대규모 좌석
            else:
                # 출구와의 거리 계산 (근접하면 1, 아니면 0)
                exit_nodes = seats_df[seats_df['type'].isin(['exit', 'door'])] if 'type' in seats_df.columns else []
                if len(exit_nodes) > 0:
                    sx, sy = row['x'], row['y']
                    min_dist = min(
                        math.hypot(sx - ex, sy - ey)
                        for _, (ex, ey) in zip(exit_nodes['x'], exit_nodes['y'])
                    )
                    crowd_type = 1 if min_dist <= 6000 else 0
                else:
                    crowd_type = 0
            current_people_ratio = local_rng.uniform(0.7, 1.0)
            
            G.add_node(row['node'],
                      x=row['x'], y=row['y'], floor=row['floor'],
                      node_count=row['node_count'], capacity=row['node_count'],
                      current_people=int(row['node_count'] * current_people_ratio),
                      type='seat', crowd_type=crowd_type, fire_arrival_time=float('inf'))
        
        # 통로 노드 추가 (deterministic하게)
        for idx, row in aisle_df.iterrows():
            # 노드별로 고정된 시드 사용
            node_seed = hash(row['aisle_id']) % (2**32)
            local_rng = np.random.RandomState(node_seed)
            
            current_people_ratio = local_rng.uniform(0.1, 0.4)
            
            G.add_node(row['aisle_id'],
                      x=row['x'], y=row['y'], floor=row['floor'],
                      node_count=0, capacity=row['capacity'],
                      current_people=int(row['capacity'] * current_people_ratio),
                      type='aisle', crowd_type=-1, fire_arrival_time=float('inf'))
        
        # 출구 노드 추가
        for idx, row in exit_df.iterrows():
            G.add_node(row['node_id'],
                      x=row['x'], y=row['y'], floor=row['floor'],
                      node_count=0, capacity=row['capacity'], exit_capacity=row['capacity'],
                      current_people=0, type='exit', crowd_type=-1, fire_arrival_time=float('inf'))
        
        # 엣지 추가
        for idx, row in edges_df.iterrows():
            G.add_edge(row['u'], row['v'], weight=row['dist'])
        
        from model import get_node_features
        node_features, node_list = get_node_features(G)
        
        # 노드 데이터프레임 생성
        nodes_data = []
        for node in sorted(G.nodes()):  # 정렬하여 순서 고정
            node_attrs = G.nodes[node]
            nodes_data.append({
                'id': node, 'x': node_attrs['x'], 'y': node_attrs['y'],
                'floor': node_attrs['floor'], 'type': node_attrs['type'],
                'capacity': node_attrs['capacity'],
                'current_people': node_attrs.get('current_people', 0),
                'crowd_type': node_attrs.get('crowd_type', -1),
                'fire_arrival_time': node_attrs.get('fire_arrival_time', float('inf'))
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        
        # 출구 정보 생성 (deterministic하게)
        exits = nodes_df[nodes_df['type'] == 'exit'].copy()
        exits['name'] = exits['id']
        
        # 출구별로 고정된 시드 사용
        exit_flows = []
        for exit_id, cap in zip(exits['id'], exits['capacity']):
            exit_seed = hash(exit_id) % (2**32)
            local_rng = np.random.RandomState(exit_seed)
            flow_ratio = local_rng.uniform(0.3, 0.6)
            exit_flows.append(int(cap * flow_ratio))
        
        exits['current_flow'] = exit_flows
        exits['status'] = 'open'
        
        return G, nodes_df, exits, edges_df, seats_df, aisle_df, node_features, node_list
        
    except FileNotFoundError as e:
        st.error(f"CSV 파일을 찾을 수 없습니다: {e}")
        st.info(f"필요한 파일: SeatNodes_{suffix}.csv, AisleNodes_{suffix}.csv, ExitNodes_{suffix}.csv, Edges_{suffix}_withXY.csv")
        return None, None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None, None, None, None, None


def load_seat_units_csv(venue_type):
    """개별 좌석 CSV를 읽어 전처리한 DataFrame을 반환
    - 세션 저장은 main에서 호출 시점에 수행한다.
    - venue_type: 'I형' 또는 'T형'
    """
    import pandas as pd
    
    # venue_type에 따라 경로 결정
    if venue_type == 'I형':
        path = "data/Seat_Node_mapped_I.csv"
    else:  # T형
        path = "data/Seat_Node_mapped_T.csv"
    
    try:
        df = pd.read_csv(path)
        df['seat_id'] = df['seat_id'].astype(str).str.strip()
        df['node'] = df['node'].astype(str).str.strip()
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna(subset=['seat_id', 'node', 'x', 'y'])
        return df
    except Exception as e:
        st.warning(f"개별 좌석 CSV 로드 실패 ({venue_type}): {e}")
        return None