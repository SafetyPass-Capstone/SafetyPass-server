"""FastAPI 서버 - Streamlit 상태 관리 및 프론트엔드 API"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from redis_manager import get_redis_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="대피 시뮬레이션 API",
    description="Streamlit 대피 시뮬레이션 상태 관리 및 제어 API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_mgr = get_redis_manager()


class StateUpdate(BaseModel):
    session_id: str
    key: str
    value: Any


class CommandRequest(BaseModel):
    session_id: str
    action: str
    params: Optional[Dict[str, Any]] = None


class SimulationStartRequest(BaseModel):
    fire_node: int
    venue_type: Optional[str] = "I"


class SimulationControlRequest(BaseModel):
    action: str


@app.get("/")
async def root():
    """API 루트"""
    return {
        "service": "Evacuation Simulation API",
        "version": "1.0.0",
        "status": "running",
        "redis_connected": redis_mgr.is_connected()
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    redis_status = redis_mgr.is_connected()
    return {
        "status": "healthy" if redis_status else "degraded",
        "redis": "connected" if redis_status else "disconnected"
    }


@app.post("/api/state/set")
async def set_state(data: StateUpdate):
    """Streamlit 상태를 Redis에 저장"""
    try:
        success = redis_mgr.save_state(data.session_id, data.key, data.value)
        if success:
            return {"status": "success", "message": "State saved"}
        else:
            raise HTTPException(status_code=503, detail="Redis connection failed")
    except Exception as e:
        logger.error(f"상태 저장 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/state/get/{session_id}/{key}")
async def get_state(session_id: str, key: str):
    """특정 상태 조회"""
    try:
        value = redis_mgr.get_state(session_id, key)
        return {"status": "success", "data": value}
    except Exception as e:
        logger.error(f"상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/state/all/{session_id}")
async def get_all_states(session_id: str):
    """세션의 모든 상태 조회"""
    try:
        data = redis_mgr.get_all_states(session_id)
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"전체 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/state/delete/{session_id}/{key}")
async def delete_state(session_id: str, key: str):
    """상태 삭제"""
    try:
        success = redis_mgr.delete_state(session_id, key)
        if success:
            return {"status": "success", "message": "State deleted"}
        else:
            raise HTTPException(status_code=503, detail="Redis connection failed")
    except Exception as e:
        logger.error(f"상태 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/simulation/status/{session_id}")
async def get_simulation_status(session_id: str):
    """시뮬레이션 상태 조회"""
    try:
        status = {
            "simulation_running": redis_mgr.get_state(session_id, "simulation_running"),
            "current_time": redis_mgr.get_state(session_id, "current_time"),
            "fire_node": redis_mgr.get_state(session_id, "fire_node"),
            "venue_type": redis_mgr.get_state(session_id, "venue_type"),
            "evacuated_count": redis_mgr.get_state(session_id, "evacuated_count"),
            "total_people": redis_mgr.get_state(session_id, "total_people")
        }
        return {"status": "success", "data": status}
    except Exception as e:
        logger.error(f"시뮬레이션 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulation/start/{session_id}")
async def start_simulation(session_id: str, request: SimulationStartRequest):
    """시뮬레이션 시작 명령"""
    try:
        command = {
            "action": "start",
            "fire_node": request.fire_node,
            "venue_type": request.venue_type
        }
        success = redis_mgr.save_command(session_id, command)
        if success:
            return {"status": "success", "message": "Start command queued"}
        else:
            raise HTTPException(status_code=503, detail="Redis connection failed")
    except Exception as e:
        logger.error(f"시뮬레이션 시작 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulation/control/{session_id}")
async def control_simulation(session_id: str, request: SimulationControlRequest):
    """시뮬레이션 제어 (pause/resume/stop)"""
    try:
        command = {"action": request.action}
        success = redis_mgr.save_command(session_id, command)
        if success:
            return {"status": "success", "message": f"{request.action} command queued"}
        else:
            raise HTTPException(status_code=503, detail="Redis connection failed")
    except Exception as e:
        logger.error(f"시뮬레이션 제어 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/output/save/{session_id}")
async def save_output(session_id: str, output_data: Dict[str, Any]):
    """화면 출력 결과 저장"""
    try:
        success = redis_mgr.save_output(session_id, output_data)
        if success:
            return {"status": "success", "message": "Output saved"}
        else:
            raise HTTPException(status_code=503, detail="Redis connection failed")
    except Exception as e:
        logger.error(f"출력 저장 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/output/get/{session_id}")
async def get_output(session_id: str):
    """저장된 화면 출력 조회"""
    try:
        data = redis_mgr.get_output(session_id)
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"출력 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evacuation/paths/{session_id}")
async def get_evacuation_paths(session_id: str):
    """대피 경로 데이터 조회"""
    try:
        paths = redis_mgr.get_state(session_id, "evacuation_paths")
        pred_exit = redis_mgr.get_state(session_id, "pred_exit")
        total_time = redis_mgr.get_state(session_id, "total_evacuation_time")
        avg_time = redis_mgr.get_state(session_id, "avg_evacuation_time")
        
        return {
            "status": "success",
            "data": {
                "paths": paths,
                "predicted_exits": pred_exit,
                "total_evacuation_time": total_time,
                "average_evacuation_time": avg_time
            }
        }
    except Exception as e:
        logger.error(f"대피 경로 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evacuation/bottleneck/{session_id}")
async def get_bottleneck(session_id: str):
    """병목 지점 데이터 조회"""
    try:
        bottleneck = redis_mgr.get_state(session_id, "bottleneck_data")
        return {"status": "success", "data": bottleneck}
    except Exception as e:
        logger.error(f"병목 데이터 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fire/spread/{session_id}")
async def get_fire_spread(session_id: str):
    """화재 확산 데이터 조회"""
    try:
        fire_data = redis_mgr.get_state(session_id, "fire_spread_data")
        return {"status": "success", "data": fire_data}
    except Exception as e:
        logger.error(f"화재 데이터 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/command/send")
async def send_command(request: CommandRequest):
    """일반 명령 전송"""
    try:
        command = {
            "action": request.action,
            "params": request.params or {}
        }
        success = redis_mgr.save_command(request.session_id, command)
        if success:
            return {"status": "success", "message": "Command queued"}
        else:
            raise HTTPException(status_code=503, detail="Redis connection failed")
    except Exception as e:
        logger.error(f"명령 전송 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== 공유 세션 설정 ==========
SHARED_SESSION_ID = "evacuation-system-shared"
# ===================================

# ========== 프론트엔드 전용 API (세션 ID 불필요) ==========

@app.get("/api/frontend/emergency-status")
async def get_emergency_status():
    """
    긴급 상황 여부 확인
    - 화재 발생 여부만 반환 (간소화)
    """
    try:
        session_id = SHARED_SESSION_ID
        simulation_running = redis_mgr.get_state(session_id, "simulation_running")
        fire_node = redis_mgr.get_state(session_id, "fire_node")
        
        is_emergency = simulation_running and fire_node is not None
        
        return {
            "status": "success",
            "data": {
                "is_emergency": is_emergency,
                "fire_node": fire_node
            }
        }
    except Exception as e:
        logger.error(f"긴급 상황 확인 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_edge_weight(graph_edges: dict, node1, node2) -> float:
    """
    엣지 가중치 딕셔너리에서 두 노드 간 거리 조회
    sidebar.py와 동일한 방식
    """
    # 다양한 키 형식 시도
    node1_str = str(node1)
    node2_str = str(node2)
    
    # 문자열 키 시도 (쉼표 구분)
    key1 = f"{node1_str},{node2_str}"
    key2 = f"{node2_str},{node1_str}"
    
    if key1 in graph_edges:
        return float(graph_edges[key1])
    if key2 in graph_edges:
        return float(graph_edges[key2])
    
    # 언더스코어 구분 시도
    key3 = f"{node1_str}_{node2_str}"
    key4 = f"{node2_str}_{node1_str}"
    
    if key3 in graph_edges:
        return float(graph_edges[key3])
    if key4 in graph_edges:
        return float(graph_edges[key4])
    
    # 정수형 노드 시도
    try:
        node1_int = int(node1) if not isinstance(node1, int) else node1
        node2_int = int(node2) if not isinstance(node2, int) else node2
        
        key5 = f"{node1_int},{node2_int}"
        key6 = f"{node2_int},{node1_int}"
        
        if key5 in graph_edges:
            return float(graph_edges[key5])
        if key6 in graph_edges:
            return float(graph_edges[key6])
    except:
        pass
    
    logger.warning(f"엣지 가중치를 찾을 수 없음: {node1} -> {node2}")
    return 0.0


@app.get("/api/frontend/evacuation-info/{seat_id}")
async def get_evacuation_info_for_seat(seat_id: str):
    """
    특정 좌석의 대피 정보 조회
    sidebar.py의 render_evacuation_path_selector()와 동일한 계산 방식 사용
    """
    try:
        session_id = SHARED_SESSION_ID
        
        # 긴급 상황 확인
        simulation_running = redis_mgr.get_state(session_id, "simulation_running")
        fire_node = redis_mgr.get_state(session_id, "fire_node")
        
        if not simulation_running or fire_node is None:
            return {
                "status": "success",
                "data": {
                    "is_emergency": False,
                    "message": "긴급 상황이 아닙니다"
                }
            }
        
        # 대피 경로 정보
        evacuation_paths = redis_mgr.get_state(session_id, "evacuation_paths")
        
        if not evacuation_paths or seat_id not in evacuation_paths:
            logger.error(f"좌석 {seat_id}의 대피 경로를 찾을 수 없습니다")
            logger.error(f"사용 가능한 좌석: {list(evacuation_paths.keys())[:5] if evacuation_paths else 'None'}")
            return {
                "status": "error",
                "message": f"좌석 {seat_id}의 대피 경로를 찾을 수 없습니다"
            }
        
        path = evacuation_paths[seat_id]
        if not path or len(path) < 2:
            logger.error(f"좌석 {seat_id}의 유효한 경로가 없습니다: {path}")
            return {
                "status": "error",
                "message": f"좌석 {seat_id}의 유효한 경로가 없습니다"
            }
        
        optimal_exit = path[-1]
        logger.info(f"좌석 {seat_id}의 경로: {path[:3]}...{path[-3:]} (총 {len(path)}개 노드)")
        
        # ========== sidebar.py와 동일한 시간 계산 ==========
        graph_edges = redis_mgr.get_state(session_id, "graph_edges")
        
        if graph_edges and isinstance(graph_edges, dict):
            logger.info(f"그래프 엣지 데이터 존재: {len(graph_edges)}개")
            
            # 이 거리 계산 (sidebar.py와 동일)
            total_distance = 0.0
            
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                
                # 엣지 가중치 조회
                weight = get_edge_weight(graph_edges, current_node, next_node)
                
                if weight > 0:
                    total_distance += weight
                    logger.debug(f"구간 {current_node}->{next_node}: {weight:.2f}mm")
                else:
                    logger.warning(f"구간 {current_node}->{next_node}: 가중치 없음")
            
            logger.info(f"좌석 {seat_id}: 총 거리 = {total_distance:.1f}mm")
            
            # 예상 대피 시간 계산 (sidebar.py와 동일)
            avg_speed = 1300  # mm/s
            estimated_time_seconds = total_distance / avg_speed if avg_speed > 0 else 0.0
            
            logger.info(f"좌석 {seat_id}: 예상 시간 = {estimated_time_seconds:.1f}초")
            
        else:
            # graph_edges가 없으면 fallback
            logger.warning(f"그래프 엣지 데이터 없음. 평균값 사용")
            avg_evacuation_time = redis_mgr.get_state(session_id, "avg_evacuation_time")
            if avg_evacuation_time:
                estimated_time_seconds = float(avg_evacuation_time)
            else:
                # 최후의 fallback: 노드 수 기반 추정
                estimated_time_seconds = len(path) * 2.0
        
        # 최소 시간 보장
        if estimated_time_seconds < 1:
            logger.warning(f"시간이 0초. 경로 길이 기반 추정 사용")
            estimated_time_seconds = len(path) * 1.5
        
        # 분/초 변환 (sidebar.py와 동일)
        minutes = int(estimated_time_seconds // 60)
        seconds = int(estimated_time_seconds % 60)
        estimated_time = f"{minutes}분 {seconds}초"
        
        logger.info(f"최종 결과 - 좌석 {seat_id}: {estimated_time}")
        # ==============================================
        
        # 닫힌 출구 목록
        exit_status = redis_mgr.get_state(session_id, "exit_status") or {}
        closed_exits = [exit_name for exit_name, is_open in exit_status.items() if not is_open]
        
        # 응답 데이터
        response_data = {
            "is_emergency": True,
            "seat_id": seat_id,
            "optimal_exit": optimal_exit,
            "estimated_time": estimated_time,  # "1분 30초" 형태
            "fire_location": fire_node,
            "closed_exits": closed_exits,
            "evacuation_path": path,
            "path_length": len(path),  # 경로 노드 수
            "total_distance_mm": total_distance if 'total_distance' in locals() else 0  # 총 거리
        }
        
        return {
            "status": "success",
            "data": response_data
        }
    
    except Exception as e:
        logger.error(f"대피 정보 조회 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/frontend/closed-exits")
async def get_closed_exits():
    """닫힌 출구 목록 조회"""
    try:
        session_id = SHARED_SESSION_ID
        exit_status = redis_mgr.get_state(session_id, "exit_status") or {}
        closed_exits = [exit_name for exit_name, is_open in exit_status.items() if not is_open]
        
        return {
            "status": "success",
            "data": {
                "closed_exits": closed_exits,
                "total_exits": len(exit_status),
                "active_exits": len(exit_status) - len(closed_exits)
            }
        }
    except Exception as e:
        logger.error(f"닫힌 출구 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/frontend/all-seats")
async def get_all_seats():
    """모든 좌석 ID 목록 조회"""
    try:
        session_id = SHARED_SESSION_ID
        evacuation_paths = redis_mgr.get_state(session_id, "evacuation_paths") or {}
        seat_ids = list(evacuation_paths.keys())
        
        return {
            "status": "success",
            "data": {
                "seat_ids": seat_ids,
                "total_seats": len(seat_ids)
            }
        }
    except Exception as e:
        logger.error(f"좌석 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)