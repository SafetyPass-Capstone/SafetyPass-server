# visualization.py
"""그래프 시각화 함수"""

import plotly.graph_objects as go
import numpy as np
from utils import get_bottleneck_color
from config import CROWD_TYPE_COLORS, CROWD_TYPE_NAMES
import json

def load_geojson(path):
    """GeoJSON 파일 로드 (경로 기본값: 업로드된 파일 위치)"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print("GeoJSON load error:", e)
        return None

def polygon_ring_to_path(ring, transform=lambda x, y: (x, y)):
    """
    ring: [[x,y], [x,y], ...]
    transform: 좌표 변환 함수 (여기선 아이덴티티)
    반환: Plotly path string (SVG 스타일)
    """
    pts = [transform(x, y) for x, y in ring]
    # 보장: 닫힌 경로
    if pts[0] != pts[-1]:
        pts = pts + [pts[0]]
    path = "M " + " L ".join(f"{x},{y}" for x, y in pts) + " Z"
    return path

def add_geojson_layer_as_shapes(
    fig, geojson, venue_type='I형', layer_filter=None,
    default_fill='rgba(235, 235, 235, 0.3)',
    default_line='rgba(100,100,100,0.6)',
    layer_property_name='layer'
):
    """
    GeoJSON의 Polygon / MultiPolygon feature들을 plotly shape(path)로 추가.
    - stage 레이어는 #a09dd0 색으로 채우고 중앙에 'Stage' 텍스트 추가
    - venue_type: 'I형' 또는 'T형'
    """
    if not geojson:
        return

    # venue_type에 따라 stage 레이어 이름 결정
    stage_layer = "stage_I" if venue_type == 'I형' else "stage_T"

    features = geojson.get('features', [])
    for feat in features:
        props = feat.get('properties', {}) or {}
        layer_name = props.get(layer_property_name, None)

        # 필터링
        if layer_filter is not None:
            if isinstance(layer_filter, str):
                if layer_name != layer_filter:
                    continue
            elif callable(layer_filter):
                if not layer_filter(props):
                    continue

        geom = feat.get('geometry', {})
        gtype = geom.get('type')
        coords = geom.get('coordinates', [])

        # 지원되는 지오메트리만 처리
        polygons = []
        if gtype == 'Polygon':
            polygons = [coords]
        elif gtype == 'MultiPolygon':
            polygons = coords
        else:
            continue

        # 각 폴리곤의 외곽 링(첫번째 링)을 path로 변환하여 shape로 추가
        for poly in polygons:
            if not poly or not poly[0]:
                continue
            outer_ring = poly[0]  # 외곽 링
            path = polygon_ring_to_path(outer_ring)

            # 기본 색상 지정
            fillcolor = props.get('fill', default_fill)
            linecolor = props.get('line', default_line)

            # 특정 레이어(stage)는 색상 지정
            if layer_name == stage_layer:
                fillcolor = "#9d9d9d"

            fig.add_shape(
                type="path",
                path=path,
                fillcolor=fillcolor,
                line=dict(color=linecolor, width=0.8),
                layer="below"
            )

            # stage 레이어에만 중앙 텍스트 추가
            if layer_name == stage_layer:
                # 중심 좌표 계산
                xs = [p[0] for p in outer_ring]
                ys = [p[1] for p in outer_ring]
                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)

                fig.add_trace(go.Scatter(
                    x=[center_x+1000], y=[center_y],
                    mode="text",
                    text=["Stage"],
                    textfont=dict(size=13, color="#f2f2f2", family="Arial Black"),
                    hoverinfo="none",
                    showlegend=False
                ))

def create_graph_figure(G, mode, color_mode, st_session_state, venue_type='I형'):
    """그래프 시각화 생성"""
    fig = go.Figure()

    # --- GeoJSON 배경(성능 우선 shapes 방식) ---
    if venue_type == 'I형':
        geojson_path = "data/I_Layer_geometry.json"
        cache_key = 'I_layer_geojson'
    else:  # T형
        geojson_path = "data/T_Layer_geometry.json"
        cache_key = 'T_layer_geojson'
    
    # 캐시 로드
    geojson = getattr(st_session_state, cache_key, None) if hasattr(st_session_state, cache_key) else None
    if geojson is None:
        geojson = load_geojson(geojson_path)
        try:
            setattr(st_session_state, cache_key, geojson)
        except Exception:
            pass

    add_geojson_layer_as_shapes(fig, geojson, venue_type)

    # 엣지 그리기
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['x'], G.nodes[edge[0]]['y']
        x1, y1 = G.nodes[edge[1]]['x'], G.nodes[edge[1]]['y']
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color='#898989', width=0.7),
            hoverinfo='none',
            showlegend=False
        ))
    
    # 노드 그리기 (화재 모드 + 병목도 토글 ON일 때만)
    if mode == "화재 모드" and color_mode == "병목도":
        add_bottleneck_nodes(fig, G)
        add_group_seat_nodes_dim(fig, G)  # 좌석 묶음 흐림 레이어

    # 좌석(개별) 점 레이어 - 세션에 존재할 때만 표시
    add_seat_unit_points(fig, st_session_state)
    
    # 출구 노드
    add_exit_nodes(fig, G, st_session_state)

    # 화재 모드일 때 추가적인 레이어
    if mode == "화재 모드":
        add_fire_mode_elements(fig, G, st_session_state)

    # 병목도 히트맵 범례 추가 (더미 trace 사용)
    # 구분선
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='#30363d', width=0),
        name=' ',
        showlegend=True,
        hoverinfo='none'
    ))

    # 제목
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(width=0),
        name='<b>병목도 히트맵</b>',
        showlegend=True,
        hoverinfo='none'
    ))

    # 높음 / 중간 / 낮음 3단계
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=8, color='#FF1A1A'),  # 빨강
        name='높음',
        showlegend=True,
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=8, color='#D966FF'),  # 연보라
        name='중간',
        showlegend=True,
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=8, color='#338DFF'),  # 파랑
        name='낮음',
        showlegend=True,
        hoverinfo='none'
    ))
    
    # 레이아웃 설정
    title_suffix = get_title_suffix(mode, color_mode, st_session_state)
    
    fig.update_layout(
        title=dict(
            text=f"공연장 그래프 ({venue_type}) - {mode}{title_suffix}",
            font=dict(size=18, color='#ffffff')
        ),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        showlegend=True,
        legend=dict(
            bgcolor='rgba(22, 27, 34, 0.8)',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(color='#c9d1d9')
        ),
        hovermode='closest',
        height=600,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig


def add_bottleneck_nodes(fig, G):
    """병목도 기반 노드 시각화 (3단계: 높음/중간/낮음)"""
    from config import CROWD_TYPE_NAMES

    # 레벨별 버킷
    buckets = {
        "high": {"x": [], "y": [], "text": []},  # 빨강
        "mid":  {"x": [], "y": [], "text": []},  # 연보라
        "low":  {"x": [], "y": [], "text": []},  # 파랑
    }

    def classify_level(bval: float):
        # 병목도(bottleneck = demand / capacity)가 얼마나 큰지에 따라 색 레벨 결정
        if bval >= 0.10:
            return "high"
        elif bval >= 0.05:
            return "mid"
        else:
            return "low"

    # 'aisle' 노드와 'seat' 노드만 시각화 대상으로 삼는다
    for node_type in ['aisle', 'seat']:
        # 먼저 해당 타입 노드 목록만 뽑음 (여기가 핵심 수정: 리스트 컴프리헨션에서 바로 돌지 않기)
        type_nodes = [nn for nn in G.nodes if G.nodes[nn].get('type') == node_type]

        for node_id in type_nodes:
            nd = G.nodes[node_id]

            x = nd.get('x', None)
            y = nd.get('y', None)
            if x is None or y is None:
                continue  # 좌표 없으면 표시 못 하니까 스킵

            bottleneck = float(nd.get('bottleneck', 0.0))

            # crowd type 이름(hover용) - 방어적으로 처리
            crowd_code = nd.get('crowd_type', 1)
            if crowd_code not in [0, 1, 2]:
                crowd_code = 1
            crowd_name = (
                CROWD_TYPE_NAMES[crowd_code]
                if (hasattr(CROWD_TYPE_NAMES, "__getitem__") and crowd_code in CROWD_TYPE_NAMES)
                else f"유형{crowd_code}"
            )

            demand_people = nd.get('demand_people', 'N/A')
            cap_used = nd.get('node_capacity_used_for_bottleneck',
                              nd.get('capacity', 'N/A'))

            hover_txt = (
                f"{node_id}<br>"
                f"군중 유형: {crowd_name}<br>"
            )

            level = classify_level(bottleneck)

            buckets[level]["x"].append(x)
            buckets[level]["y"].append(y)
            buckets[level]["text"].append(hover_txt)

    # 이제 각 버킷별 점 trace를 fig에 추가
    # 높음 (빨강)
    fig.add_trace(go.Scatter(
        x=buckets["high"]["x"],
        y=buckets["high"]["y"],
        mode='markers',
        marker=dict(size=8, color='#FF1A1A'),
        text=buckets["high"]["text"],
        hoverinfo='text',
        name='높음',
        showlegend=False  # 실제 범례는 create_graph_figure에서 dummy trace로 넣고 있으니 False
    ))

    # 중간 (연보라)
    fig.add_trace(go.Scatter(
        x=buckets["mid"]["x"],
        y=buckets["mid"]["y"],
        mode='markers',
        marker=dict(size=8, color='#D966FF'),
        text=buckets["mid"]["text"],
        hoverinfo='text',
        name='중간',
        showlegend=False
    ))

    # 낮음 (파랑)
    fig.add_trace(go.Scatter(
        x=buckets["low"]["x"],
        y=buckets["low"]["y"],
        mode='markers',
        marker=dict(size=8, color='#338DFF'),
        text=buckets["low"]["text"],
        hoverinfo='text',
        name='낮음',
        showlegend=False
    ))



def add_crowd_type_nodes(fig, G):
    """군중 유형 기반 노드 추가"""
    for node_type in ['aisle', 'seat']:
        nodes = [n for n in G.nodes if G.nodes[n]['type'] == node_type]
        x_coords = [G.nodes[n]['x'] for n in nodes]
        y_coords = [G.nodes[n]['y'] for n in nodes]
        
        # 직접 처리
        colors = []
        texts = []
        for n in nodes:
            crowd_type = G.nodes[n].get('crowd_type', 1)
            if crowd_type not in [0, 1, 2]:
                crowd_type = 1
            colors.append(CROWD_TYPE_COLORS[crowd_type])
            texts.append(f"{n}<br>유형: {CROWD_TYPE_NAMES[crowd_type]}")
        
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='markers',
            marker=dict(size=5 if node_type == 'seat' else 3, color=colors),
            text=texts,
            hoverinfo='text',
            name=node_type,
            showlegend=False
        ))

def add_exit_nodes(fig, G, st_session_state):
    """출구 노드 추가 (둥근 사각형 + 그림자 + 중앙 텍스트)
       색상은 st_session_state['exit_status'][name] 또는 개별 toggle key에 따라 결정
    """
    exit_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'exit']
    # try to get central dict
    exit_status_dict = {}
    try:
        exit_status_dict = st_session_state.get('exit_status', {}) or {}
    except Exception:
        exit_status_dict = {}

    for n in exit_nodes:
        x = G.nodes[n]['x']
        y = G.nodes[n]['y']
        name = str(n)

        # 상태 결정 우선순위:
        # 1) exit_status_dict[name]  2) st_session_state[f"exit_toggle_{name}"]  3) True
        is_open = True
        if name in exit_status_dict:
            is_open = bool(exit_status_dict[name])
        else:
            key = f"exit_toggle_{name}"
            if key in st_session_state:
                is_open = bool(st_session_state.get(key, True))
            else:
                is_open = True

        # 색상 결정
        main_fill = "#34ee4d" if is_open else "#ff4d4f"
        shadow_rgb = (87, 244, 54) if is_open else (255, 77, 79)

        # (이하 기존 rounded rect + shadow 추가 코드 사용)
        w, h, r = 5000, 2400, 700
        x0, y0 = x - w/2, y - h/2
        x1, y1 = x + w/2, y + h/2

        def rounded_rect_path(x0, y0, x1, y1, r):
            return (
                f"M {x0+r},{y0} "
                f"L {x1-r},{y0} Q {x1},{y0} {x1},{y0+r} "
                f"L {x1},{y1-r} Q {x1},{y1} {x1-r},{y1} "
                f"L {x0+r},{y1} Q {x0},{y1} {x0},{y1-r} "
                f"L {x0},{y0+r} Q {x0},{y0} {x0+r},{y0} Z"
            )

        # main shape
        fig.add_shape(
            type="path",
            path=rounded_rect_path(x0, y0, x1, y1, r),
            fillcolor=main_fill,
            line=dict(color=main_fill, width=0),
            layer="below"
        )

        # shadows
        for i in range(5):
            offset = 120 * (i + 1)
            radius = 800 + 80 * i
            alpha = 0.35 / (i + 1)
            r0, g0, b0 = shadow_rgb
            fill_rgba = f"rgba({r0},{g0},{b0},{alpha})"
            fig.add_shape(
                type="path",
                path=rounded_rect_path(x0 - offset, y0 - offset, x1 + offset, y1 + offset, radius),
                fillcolor=fill_rgba,
                layer="below",
                line=dict(width=0),
            )

        # center text
        text_color = "#06120a" if is_open else "#ffffff"
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            text=[f"<b>{name}</b>"],
            mode="text",
            textfont=dict(family="Arial Black", size=13, color=text_color),
            hovertext=[f"출구: {name}"],
            hoverinfo="text",
            showlegend=False,
            name="출구"
        ))

    fig.update_yaxes(scaleanchor="x", scaleratio=1)


def add_group_seat_nodes_dim(fig, G):
    """좌석 노드(묶음)를 흐린 네모로 표시"""
    group_nodes = [n for n in G.nodes if G.nodes[n]['type'] == 'seat']
    if not group_nodes:
        return
    x_coords = [G.nodes[n]['x'] for n in group_nodes]
    y_coords = [G.nodes[n]['y'] for n in group_nodes]
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers',
        marker=dict(
            size=20,
            color='rgba(200,200,200,0.25)',
            symbol='square',
        ),
        text=[f"좌석 노드: {n}" for n in group_nodes],
        hoverinfo='skip',
        name='좌석 노드(묶음)',
        showlegend=True
    ))

def add_seat_unit_points(fig, st_session_state):
    """개별 좌석을 점으로 표시. 경로 계산에는 영향 없음."""
    df = getattr(st_session_state, 'seat_units_df', None)
    if df is None or df.empty:
        return
    x_coords = df['x'].tolist()
    y_coords = df['y'].tolist()
    texts = [f"좌석: {sid}<br>노드: {node}" for sid, node in zip(df['seat_id'], df['node'])]
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers',
        marker=dict(
            size=2.5,
            color='#9d886d',
            symbol='circle',
            line=dict(color='#9d886d', width=0.5)
        ),
        text=texts,
        hoverinfo='skip',
        name='좌석(개별)',
        showlegend=True
    ))



def add_fire_mode_elements(fig, G, st_session_state):
    """화재 모드 요소 추가"""
    # 선택된 좌석의 경로 표시
    if 'selected_seat_fire' in st_session_state and st_session_state.selected_seat_fire != "선택 안함":
        selected_seat = st_session_state.selected_seat_fire
        
        if selected_seat in st_session_state.evacuation_paths:
            path = st_session_state.evacuation_paths[selected_seat]
            if path:
                # 경로 선 그리기
                for i in range(len(path) - 1):
                    x0, y0 = G.nodes[path[i]]['x'], G.nodes[path[i]]['y']
                    x1, y1 = G.nodes[path[i+1]]['x'], G.nodes[path[i+1]]['y']
                    fig.add_trace(go.Scatter(
                        x=[x0, x1], y=[y0, y1],
                        mode='lines',
                        line=dict(color='#ff4546', width=4),
                        showlegend=False,
                        hoverinfo='none'
                    ))
                
                # 경로 노드들 강조
                path_x = [G.nodes[n]['x'] for n in path[1:-1]]
                path_y = [G.nodes[n]['y'] for n in path[1:-1]]
                
                if path_x:
                    fig.add_trace(go.Scatter(
                        x=path_x, y=path_y,
                        mode='markers',
                        marker=dict(size=8, color='#ff4546', line=dict(color='#FFA4A5', width=1)),
                        text=[f"경로: {n}" for n in path[1:-1]],
                        hoverinfo='text',
                        name='대피 경로',
                        showlegend=True
                    ))
                
                # 시작점 (선택된 좌석)
                fig.add_trace(go.Scatter(
                    x=[G.nodes[path[0]]['x']],
                    y=[G.nodes[path[0]]['y']],
                    mode='markers',
                    marker=dict(size=15, color='#ff4546', symbol='star', line=dict(color='#FFA4A5', width=2)),
                    text=[f"선택 좌석: {path[0]}"],
                    hoverinfo='text',
                    name='선택 좌석',
                    showlegend=True
                ))
    
    # 화재 시각화
    if st_session_state.fire_node:
        from fire_simulation import get_fire_nodes
        
        current_time_min = st_session_state.current_time / 60
        fire_nodes, fire_approaching = get_fire_nodes(G, current_time_min)
        
        # 화재 도달 노드
        if fire_nodes:
            fire_x = [G.nodes[n]['x'] for n in fire_nodes]
            fire_y = [G.nodes[n]['y'] for n in fire_nodes]
            
            fig.add_trace(go.Scatter(
                x=fire_x, y=fire_y,
                mode='markers',
                marker=dict(
                    size=12,
                    color='#ff1a1a',
                    symbol='circle',
                    line=dict(color='#ff4444', width=2)
                ),
                text=[f"화재: {n}" for n in fire_nodes],
                hoverinfo='text',
                name='화재 구역',
                showlegend=True
            ))
        
        # 화재 접근 노드
        if fire_approaching:
            approach_x = [G.nodes[n]['x'] for n in fire_approaching]
            approach_y = [G.nodes[n]['y'] for n in fire_approaching]
            
            fig.add_trace(go.Scatter(
                x=approach_x, y=approach_y,
                mode='markers',
                marker=dict(
                    size=10,
                    color='#ff8800',
                    symbol='circle',
                    line=dict(color='#ffaa00', width=1)
                ),
                text=[f"화재 접근: {n}" for n in fire_approaching],
                hoverinfo='text',
                name='화재 접근',
                showlegend=True
            ))
        
        # 화재 발생 지점 강조
        fire_origin_x = G.nodes[st_session_state.fire_node]['x']
        fire_origin_y = G.nodes[st_session_state.fire_node]['y']
        
        fig.add_trace(go.Scatter(
            x=[fire_origin_x], y=[fire_origin_y],
            mode='markers',
            marker=dict(
                size=20,
                color='#ff0000',
                symbol='x',
                line=dict(color='#ffffff', width=1)
            ),
            text=[f"화재 발생 지점: {st_session_state.fire_node}"],
            hoverinfo='text',
            name='화재 발생점',
            showlegend=True
        ))

def get_title_suffix(mode, color_mode, st_session_state):
    """제목 접미사 생성"""
    title_suffix = ""
    if mode == "화재 모드":
        if 'selected_seat_fire' in st_session_state and st_session_state.selected_seat_fire != "선택 안함":
            title_suffix = f" - {st_session_state.selected_seat_fire} 경로 표시"
    
    return title_suffix