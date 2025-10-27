# config.py
"""설정 및 상수 정의"""

# 화재 파라미터
FIRE_SPREAD_SPEED = 1900  # mm/min
FIRE_BUFFER_MARGIN = 0.5  # min

# 군중 유형 정의
CROWD_TYPE_NAMES = {
    0: "자유형",
    1: "질서형",
    2: "몰림형"
}

CROWD_TYPE_COLORS = {
    0: "#ef4444",
    1: "#3b82f6",
    2: "#10b981"
}

# 병목도 색상 (5단계)
BOTTLENECK_COLORS = {
    'very_high': '#FF1A1A',  # 20% 이상
    'high': '#FF4444',       # 10-20%
    'medium': '#FF8800',     # 5-10%
    'low': '#FFCC00',        # 2-5%
    'very_low': '#88FF00'    # 0-2%
}

