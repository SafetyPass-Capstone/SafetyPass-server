# model.py
"""GNN 모델 및 관련 함수"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from utils import add_log
import math
import random

# ============================================================
# Laplacian Positional Encoding
# ============================================================
def get_lap_pos_enc(G, k=4):
    """Laplacian Positional Encoding 계산"""
    n = G.number_of_nodes()
    # 노드 순서 고정
    node_list = sorted(list(G.nodes()))
    idx = {n_: i for i, n_ in enumerate(node_list)}
    
    rows, cols, vals = [], [], []
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        rows += [i, j]
        cols += [j, i]
        vals += [1.0, 1.0]
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).flatten()
    if n <= 1:
        return np.zeros((n, k), dtype=np.float32)
    D = sp.diags(deg)
    L = D - A
    try:
        w, v = eigsh(L.asfptype(), k=min(k+1, n-1), M=D.asfptype(), sigma=0.0, which='LM')
        if v.shape[1] > 1:
            return v[:, 1:(k+1)].astype(np.float32)
    except Exception:
        pass
    return np.zeros((n, k), dtype=np.float32)

# ============================================================
# Enhanced Feature Extraction with LapPE
# ============================================================
def get_node_features(_G, use_lap_pe=True, lap_k=4):
    """강화된 노드 특징 추출 (LapPE 포함)"""
    # 노드 순서 고정 (정렬)
    node_list = sorted(list(_G.nodes()))
    
    # 좌표 정규화
    xs = np.array([float(_G.nodes[n].get('x', 0.0)) for n in node_list])
    ys = np.array([float(_G.nodes[n].get('y', 0.0)) for n in node_list])
    
    def norm(v):
        lo, hi = np.nanmin(v), np.nanmax(v)
        return (v - lo) / max(hi - lo, 1e-6)
    
    xnorm, ynorm = norm(xs), norm(ys)
    
    # Laplacian Positional Encoding
    lap = None
    if use_lap_pe:
        # 노드 순서 고정 (정렬)
        base_order = sorted(list(_G.nodes()))
        order = {n: i for i, n in enumerate(base_order)}
        lap_full = get_lap_pos_enc(_G, k=lap_k)
        lap = np.stack([
            lap_full[order[n]] if order[n] < lap_full.shape[0] else np.zeros(lap_k, dtype=np.float32)
            for n in node_list
        ], axis=0).astype(np.float32)
    
    # 특징 추출
    type_map = {'seat': 0, 'aisle': 1, 'exit': 2}
    feats = []
    
    for i, node in enumerate(node_list):
        node_data = _G.nodes[node]
        node_type = type_map.get(node_data.get('type', 'aisle'), 1)
        
        # seat과 aisle 구분
        if node_data.get('type') == 'seat':
            node_count = float(node_data.get('node_count', node_data.get('capacity', 0.0)))
            cap = 0.0
        else:
            node_count = 0.0
            cap = float(node_data.get('capacity', node_data.get('capacity', 0.0)))
        
        cap = np.log1p(max(cap, 0.0))
        
        # 화재 시간
        fire_time = node_data.get('fire_arrival_time', float('inf'))
        if fire_time == float('inf'):
            fire_time = 0.0
        else:
            fire_time = np.log1p(max(0.0, float(fire_time)))
        
        # 차수
        deg = float(_G.degree(node))
        
        # 군중 유형 원핫 인코딩
        crowd_type = int(node_data.get('crowd_type', -1))
        crowd_one_hot = [1.0 if crowd_type == t else 0.0 for t in [0, 1, 2]]
        
        # 기본 특징: [type, node_count, log_cap, log_fire, degree, norm_x, norm_y, crowd_onehot(3)]
        base = [node_type, node_count, cap, fire_time, deg, xnorm[i], ynorm[i]] + crowd_one_hot
        
        # LapPE 추가
        if use_lap_pe and lap is not None:
            base += lap[i].tolist()
        
        feats.append(base)
    
    # 텐서 변환 및 정규화
    x = torch.tensor(feats, dtype=torch.float)
    x[torch.isinf(x)] = torch.nan
    mean = torch.nanmean(x, dim=0, keepdim=True)
    std = torch.std(torch.nan_to_num(x, nan=0.0), dim=0, keepdim=True) + 1e-8
    x = (x - mean) / std
    x = torch.nan_to_num(x, nan=0.0)
    
    return x, node_list

# ============================================================
# PyG Data Creation
# ============================================================
def create_pyg_data(_G):
    """PyG 데이터 생성"""
    node_features, node_list = get_node_features(_G, use_lap_pe=True, lap_k=4)
    
    edge_index = []
    for edge in _G.edges():
        src = node_list.index(edge[0])
        dst = node_list.index(edge[1])
        edge_index.append([src, dst])
        edge_index.append([dst, src])
    
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    pyg_data = Data(x=node_features, edge_index=edge_index)
    
    return pyg_data, node_list

# ============================================================
# GAT Encoder (Enhanced)
# ============================================================
class GATEncoder(nn.Module):
    """강화된 GAT 인코더"""
    def __init__(self, in_dim, hid=128, out=128, heads=4):
        super().__init__()
        self.g1 = GATConv(in_dim, hid, heads=heads, concat=True, dropout=0.1)
        self.g2 = GATConv(hid * heads, out, heads=1, concat=False, dropout=0.1)
    
    def forward(self, x, edge_index):
        h = F.elu(self.g1(x, edge_index))
        z = F.elu(self.g2(h, edge_index))
        return F.normalize(z, dim=-1)

# ============================================================
# Prototype Head
# ============================================================
class ProtoHead(nn.Module):
    """프로토타입 헤드 - 클러스터 중심점 학습"""
    def __init__(self, d=128, K=8):
        super().__init__()
        self.K = K
        self.prototypes = nn.Parameter(torch.empty(K, d))
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))
    
    def logits(self, z, tau=0.12):
        """프로토타입과의 유사도 계산 (temperature scaling)"""
        return (z @ self.prototypes.t()) / tau  # [N, K]

# ============================================================
# Exit Mapper
# ============================================================
class ExitMapper(nn.Module):
    """프로토타입 -> 출구 확률 행렬 학습"""
    def __init__(self, K, E):
        super().__init__()
        self.map_logits = nn.Parameter(torch.zeros(K, E))
    
    def forward(self):
        """프로토타입별 출구 확률 분포 반환"""
        return F.softmax(self.map_logits, dim=-1)  # [K, E]

# ============================================================
# Model Loading
# ============================================================
@st.cache_resource
def load_trained_model(_G, venue_type='I형'):
    """학습된 모델 로드 (홀 형태별)"""
    
    # 랜덤 시드 고정 (함수 시작 시)
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    
    # deterministic 설정
    if torch.cuda.is_available():
        torch.cuda.manual_seed(7)
        torch.cuda.manual_seed_all(7)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        # 튜플인 경우 언팩
        if isinstance(_G, tuple):
            _G = _G[0]
        
        # None 체크
        if _G is None:
            raise ValueError("Graph is None")
        
        # NetworkX Graph 확인
        if not hasattr(_G, 'nodes'):
            raise TypeError(f"Expected NetworkX Graph, got {type(_G)}")
        
        # PyG 데이터 생성
        pyg_data, node_list = create_pyg_data(_G)
        
        # 출구 노드 목록 (정렬하여 순서 고정)
        exit_nodes = sorted([n for n in _G.nodes if _G.nodes[n]['type'] == 'exit'])
        
        # 입력 차원
        in_dim = pyg_data.x.size(1)
        K = len(exit_nodes) if len(exit_nodes) > 0 else 8
        
        # 모델 초기화
        encoder = GATEncoder(in_dim=in_dim, hid=128, out=128, heads=4)
        proto_head = ProtoHead(d=128, K=K)
        exit_mapper = ExitMapper(K=K, E=len(exit_nodes)) if len(exit_nodes) > 0 else None
        
        # 홀 형태에 따른 모델 파일명
        suffix = 'I' if venue_type == 'I형' else 'T'
        model_path = f'best_model-{suffix}.pt'
        
        # 저장된 가중치 로드
        checkpoint = torch.load(model_path, map_location='cpu')
        encoder.load_state_dict(checkpoint['encoder'])
        proto_head.load_state_dict(checkpoint['head'])
        if exit_mapper is not None and 'exit_mapper' in checkpoint:
            exit_mapper.load_state_dict(checkpoint['exit_mapper'])
        
        encoder.eval()
        proto_head.eval()
        if exit_mapper is not None:
            exit_mapper.eval()
        
        # 예측 수행 (deterministic하게)
        with torch.no_grad():
            z = encoder(pyg_data.x, pyg_data.edge_index)
            logits = proto_head.logits(z, tau=0.12)
            pred = logits.argmax(dim=1)
        
        add_log(f"학습된 GNN 모델 로드 완료! ({venue_type}) 출구 수: {len(exit_nodes)}, 프로토타입 수: {K}")
        
        return encoder, proto_head, exit_mapper, pyg_data, node_list, exit_nodes, pred
    
    except FileNotFoundError:
        st.error(f"'{model_path}' 파일을 찾을 수 없습니다. 같은 폴더에 있는지 확인해주세요.")
        return None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None, None, None, None